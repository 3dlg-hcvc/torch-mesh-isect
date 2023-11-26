import os
import time
import argparse
from tqdm import tqdm
import requests
import json

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import pandas as pd
import numpy as np
import cv2

import trimesh
import pyrender
from pyrender import RenderFlags
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer
from mesh_intersection.bvh_search_tree import BVH

device = torch.device('cuda')

os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

wayfair_mesh_file_path = "/datasets/internal/models3d/wayfair/wayfair_models_cleaned/{object_name}/{object_name}.glb"
threedw_mesh_file_path = "/project/3dlg-hcvc/rlsd/data/3dw/objmeshes/{object_name}/{object_name}.obj"

complete_task_list_file = '/project/3dlg-hcvc/rlsd/data/annotations/completed_tasks_latest.txt'
tasks_url = 'https://aspis.cmpt.sfu.ca/rlsd/api/scene-manager/tasks'
task_detail_url = 'https://aspis.cmpt.sfu.ca/rlsd/api/scene-manager/tasks/{task_id}/json'

task_pano_mapping = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/task_pano_mapping.json"))
task_ids = list(task_pano_mapping.keys())
task_file = "/project/3dlg-hcvc/rlsd/data/annotations/complete_task_json/{task_id}.json"


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def normalize_verts(verts, scale_along_diagonal=True):
    # centering and normalization
    min_vert, _ = torch.min(verts, 0)
    max_vert, _ = torch.max(verts, 0)
    center = (min_vert + max_vert) / 2.0
    verts -= center
    if scale_along_diagonal:
        scale = 1.0 / torch.linalg.vector_norm(max_vert - min_vert)
    else:
        scale = 1.0 / (max_vert - min_vert) # normalize each dim to 1
    verts *= scale
    return verts

def center_scene(verts):
    # centering
    min_vert, _ = torch.min(verts, 0)
    max_vert, _ = torch.max(verts, 0)
    center = (min_vert + max_vert) / 2.0
    verts -= center
    return verts

def get_rlsd_transform(scene_json):

    objects = scene_json["object"]
    obj_asset_source = scene_json["assetSource"]
    
    obj_names, obj_transforms, obj_obbs = [], [], []
    for obj in objects:
        obj_id = obj['modelId']
        obj_names.append(obj_id)
        obj_transforms.append(np.transpose(np.array(obj["transform"]["data"]).reshape(4,4)))
        
        obb = {
            "centroid": np.array(obj["obb"]["centroid"], dtype=np.float32),
            "basis": np.array(obj["obb"]["normalizedAxes"], dtype=np.float32).reshape(3, 3).T,
            "size": np.array(obj["obb"]["axesLengths"], dtype=np.float32),
        }
        if 'wayfair' in obj_id:
            obb['basis'] = obb['basis'] @ np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        obj_obbs.append(obb)

    return obj_names, obj_transforms, obj_obbs

def get_object_path(object_id):
    if "wayfair" in object_id:
        file_path = wayfair_mesh_file_path.format(object_name=object_id.split('wayfair.')[-1])
    elif "3dw" in object_id:
        file_path = threedw_mesh_file_path.format(object_name=object_id.split('3dw.')[-1])
    else:
        file_path = ""
    
    return file_path

def detect_and_plot_collisions(args):
    task_elapsed_times = [[] for _ in range(len(task_ids))]
    task_mesh1_collision_ratios = [[] for _ in range(len(task_ids))]
    count_task = 0
    
    for task_id in tqdm(task_ids):
        print(f'Processing {task_id}.')
        full_pano_id = task_pano_mapping[task_id]
        house_id, level_id, pano_id = full_pano_id.split('_')
        
        task_json = json.load(open(task_file.format(task_id=task_id)))
        scene_json = task_json["sceneJson"]["scene"]
    
        objects, mesh_transforms, obbs = get_rlsd_transform(scene_json)
        obj_meshes = []
        for ind, obj_name in enumerate(objects):
            obj_mesh_file = get_object_path(obj_name)
            obj_mesh = trimesh.load(obj_mesh_file, force="mesh", skip_materials=True)
            vertices = torch.tensor(obj_mesh.vertices, dtype=torch.float32, device=device)
            obj_mesh.vertices[:] = normalize_verts(vertices, scale_along_diagonal=False).cpu().numpy()
            
            # obj_mesh.apply_transform(mesh_transforms[ind])
            obj_mesh.vertices[:] = (obbs[ind]['basis'] @ (obj_mesh.vertices * obbs[ind]['size']).T).T + obbs[ind]['centroid']
            obj_meshes.append(obj_mesh)
    
        mesh_collisions = [[{} for _ in objects] for _ in objects]
        elapsed_times = [[] for _ in objects]
        mesh1_collision_ratios = [[] for _ in objects]
        for i, (obj1_name, mesh1) in enumerate(zip(objects, obj_meshes)):
            for j, (obj2_name, mesh2) in enumerate(zip(objects, obj_meshes)):
                
                if obj1_name == obj2_name:
                    continue
                
                num_mesh1_faces = len(mesh1.faces)
                num_mesh2_faces = len(mesh2.faces)
                
                mesh = trimesh.util.concatenate([mesh1, mesh2])
                vertices = torch.tensor(mesh.vertices,
                                        dtype=torch.float32, device=device)
                faces = torch.tensor(mesh.faces.astype(np.int64),
                                    dtype=torch.long,
                                    device=device)

                batch_size = 1
                triangles = vertices[faces].unsqueeze(dim=0)

                m = BVH(max_collisions=args.max_collisions)

                torch.cuda.synchronize()
                start = time.time()
                outputs = m(triangles)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start

                outputs = outputs.detach().cpu().numpy().squeeze()
                collisions = outputs[outputs[:, 0] >= 0, :]
                
                face_diffs = np.sign(collisions - num_mesh1_faces + 0.5)
                valid = (face_diffs[:, 0] * face_diffs[:, 1]) < 0
                collisions = collisions[valid]
                
                all_collisions = np.unique(collisions.reshape(-1))
                mesh1_collisions = all_collisions[all_collisions < num_mesh1_faces]
                mesh2_collisions = all_collisions[all_collisions >= num_mesh1_faces]

                mesh1_collision_ratio = (mesh1_collisions.shape[0] / num_mesh1_faces)
                mesh_collisions[i][j] = {
                                            'obj1_name': obj1_name,
                                            'num_mesh1_faces': num_mesh1_faces,
                                            'num_mesh1_collisions': mesh1_collisions.shape[0],
                                            'mesh1_collision_ratio': mesh1_collision_ratio,
                                            'obj2_name': obj2_name,
                                            'num_mesh2_faces': num_mesh2_faces,
                                            'num_mesh2_collisions': mesh2_collisions.shape[0],
                                            'mesh2_collision_ratio': (mesh2_collisions.shape[0] / num_mesh2_faces),
                                            'elapsed_time': elapsed_time
                                        }
                
                elapsed_times[i].append(elapsed_time)
                mesh1_collision_ratios[i].append(mesh1_collision_ratio)
                
                print(f'--------({obj1_name},{obj2_name}) complete.')
                
        with open(f"{args.output}/collisions/{task_id}.json", 'w') as f:
            json.dump({'mesh1_collision_ratios':mesh1_collision_ratios, 'elapsed_times':elapsed_times, 'mesh_collisions': mesh_collisions}, f, indent=4)
            
        task_elapsed_times[count_task].append(np.mean(np.mean(np.array(elapsed_times))))
        task_mesh1_collision_ratios[count_task].append(np.mean(np.mean(np.array(mesh1_collision_ratios))))
        count_task += 1

    return task_elapsed_times, task_mesh1_collision_ratios

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply_transform', default=False, action="store_true",
                        help='Apply transform from RLSD?')
    parser.add_argument('--max_collisions', default=16, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--output', type=str, default='./',
                        help='Save render output path')
    args, _ = parser.parse_known_args()

    elapsed_times, mesh1_collision_ratios = detect_and_plot_collisions(args)
    print('Elapsed time: {:.2f}s'.format(round(np.mean(np.array(elapsed_times))),4))
    print('Percentage of collisions = {:.4f}'.format(round(np.mean(np.array(mesh1_collision_ratios)),4)))
    print('Done')
    