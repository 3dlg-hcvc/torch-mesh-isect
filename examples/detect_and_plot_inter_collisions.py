import os
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import pandas as pd
import numpy as np
import cv2

os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

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


def detect_and_plot_collisions(mesh_file1, mesh_file2, args):
    # mesh_name = os.path.basename(mesh_file).split('.')[0]
    mesh1 = trimesh.load(mesh_file1, force="mesh", skip_materials=True)
    mesh2 = trimesh.load(mesh_file2, force="mesh", skip_materials=True)

    vertices1 = torch.tensor(mesh1.vertices,
                            dtype=torch.float32, device=device)
    mesh1.vertices[:] = normalize_verts(vertices1).cpu().numpy()
    num_mesh1_faces = len(mesh1.faces)
    
    vertices2 = torch.tensor(mesh2.vertices,
                            dtype=torch.float32, device=device)
    vertices2 = normalize_verts(vertices2).cpu().numpy()
    mesh2.vertices[:] = (vertices2 + np.array([0, 0, -0.3]))
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

    if args.save_render:
        recv_faces = mesh.faces[mesh1_collisions]
        intr_faces = mesh.faces[mesh2_collisions]

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=[0.3, 0.3, 0.3, 0.99])
        recv_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=[0.0, 0.9, 0.0, 1.0])
        intr_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=[0.9, 0.0, 0.0, 1.0])

        main_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        recv_mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(mesh.vertices, recv_faces),
            material=recv_material)
        intr_mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(mesh.vertices, intr_faces),
            material=intr_material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                            ambient_light=(0.3, 0.3, 0.3))
        scene.add(main_mesh)
        scene.add(recv_mesh)
        scene.add(intr_mesh)
    
        # Use headless rendering
        # # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        dis = 1
        azim = np.pi / 4
        elev = np.pi / 6
        cam_pose = np.eye(4)
        y = dis * np.sin(elev)
        x = dis * np.cos(elev) * np.sin(azim)
        z = dis * np.cos(elev) * np.cos(azim)
        cam_pose[:3, 3] = [x, y, z]
        rotx = np.array([
            [1.0, 0, 0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)]
        ])
        roty = np.array([
            [np.cos(azim), 0, np.sin(azim)],
            [0.0, 1, 0.0],
            [-np.sin(azim), 0, np.cos(azim)]
        ])
        cam_pose[:3, :3] = np.matmul(roty, rotx)
        scene.add(camera, pose=cam_pose)

        # Set up the light -- a single spot light in the same spot as the camera
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                    innerConeAngle=np.pi/16.0)
        scene.add(light, pose=cam_pose)

        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
        color, depth = r.render(scene)
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(os.path.join(args.output, f'inter_collisions.jpg'), color)

        if args.show:
            pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)

    return mesh1.faces.shape[0], mesh1_collisions.shape[0], mesh2.faces.shape[0],  mesh2_collisions.shape[0], elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, nargs='+',
                        help='A mesh file (.obj, .ply, etc) or mesh paths to be checked for collisions')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--show', default=False, action="store_true", 
                        help='Show visualization in pyrender Viewer')
    parser.add_argument('--save_render', default=False, action="store_true", 
                        help='Save object render and collision visualization')
    parser.add_argument('--output', type=str, default='./',
                        help='Save render output path')
    args, _ = parser.parse_known_args()

    num_mesh1_faces, num_mesh1_collisions, num_mesh2_faces, num_mesh2_collisions, elapsed_time = detect_and_plot_collisions(args.path[0], args.path[1], args)
    print('Elapsed time', elapsed_time)
    print('Number of mesh1 triangles = ', num_mesh1_faces)
    print('Number of mesh1 collisions = ', num_mesh1_collisions)
    print('Percentage of collisions (%)', num_mesh1_collisions / num_mesh1_faces)
    print('Number of mesh2 triangles = ', num_mesh2_faces)
    print('Number of mesh2 collisions = ', num_mesh2_collisions)
    print('Percentage of collisions (%)', num_mesh2_collisions / num_mesh2_faces)