# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
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


def detect_and_plot_collisions(mesh_file, args):
    mesh_name = os.path.basename(mesh_file).split('.')[0]
    mesh = trimesh.load(mesh_file, force="mesh", skip_materials=True)

    # mesh = as_mesh(obj)
    vertices = torch.tensor(mesh.vertices,
                            dtype=torch.float32, device=device)
    mesh.vertices[:] = normalize_verts(vertices).cpu().numpy()
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
    num_collisions = np.unique(collisions[:,0]).shape[0]

    if args.save_render:
        recv_faces = mesh.faces[collisions[:, 0]]
        intr_faces = mesh.faces[collisions[:, 1]]

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
        cv2.imwrite(os.path.join(args.output, f'{mesh_name}.jpg'), color)

        if args.show:
            pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)

    return (mesh_name, mesh.faces.shape[0], num_collisions, num_collisions / float(triangles.shape[1]) * 100, elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='A mesh file (.obj, .ply, etc) or mesh paths to be checked for collisions')
    parser.add_argument('--max_collisions', default=16, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--show', default=False, action="store_true", 
                        help='Show visualization in pyrender Viewer')
    parser.add_argument('--save_render', default=False, action="store_true", 
                        help='Save object render and collision visualization')
    parser.add_argument('--output', type=str, default='/project/3dlg-hcvc/rlsd/data/annotations/objects_self_collisions',
                        help='Save render output path')
    args, _ = parser.parse_known_args()

    if args.path.endswith('txt'):
        obj_paths = [p.strip() for p in open(args.path)]
        all_objs, fails = [], []
        for obj_path in tqdm(obj_paths):
            try:
                all_objs.append(detect_and_plot_collisions(obj_path, args))
            except:
                fails.append(obj_path)
        data = pd.DataFrame(all_objs, columns=['mesh_name', 'num_faces', 'num_collisions', 'collision_ratio', 'elapsed_time'])
        data.to_csv(os.path.join(args.output, "collisions.csv"), index=False)
        with open(os.path.join(args.output, "collisions_fails.txt"), "w") as f:
            for p in fails:
                f.write(f"{p}\n")
    else:
        mesh_name, num_faces, num_collisions, collision_ratio, elapsed_time = detect_and_plot_collisions(args.path, args)
        print('Number of triangles = ', num_faces)
        print('Elapsed time', elapsed_time)
        print('Number of collisions = ', num_collisions)
        print('Percentage of collisions (%)', collision_ratio)