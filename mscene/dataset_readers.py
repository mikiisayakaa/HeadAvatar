#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
from mscene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    mask_path: str
    image_name: str
    semantic_path: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None

class SceneInfo(NamedTuple):
    train_cameras: list
    val_cameras: list
    nerf_normalization: dict
    flame_info: dict
    test_cameras: Optional[list] = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        width, height = image.size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readCamerasFromTransforms(path, transformsfile):
    cam_infos = []
    flame_info = {}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]
            
        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            timestep = frame["timestep_index"]
            camera_id = frame["camera_index"]
            file_path = frame["file_path"]
            image_name = os.path.basename(file_path)
            flame_name = image_name.split("_")[0] + ".npz"
            data_path = os.path.dirname(os.path.dirname(os.path.join(path, file_path)))
            semantic_path = os.path.join(data_path, "semantic", image_name)
            if not os.path.exists(semantic_path):
                continue
            flame_path = os.path.join(data_path, "flame_param", flame_name)
            flame_info[timestep] = flame_path
            mask_path = os.path.join(data_path, "fg_masks", image_name)
            image_path = os.path.join(data_path, "images", image_name)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([0, 0, 0])
            
            if 'w' in frame and 'h' in frame:
                image = None
                width = frame['w']
                height = frame['h']
            else:
                raise RuntimeError("Image width and height must be specified in transforms.json")

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image, 
                image_path=image_path, image_name=image_name, 
                mask_path=mask_path, semantic_path=semantic_path,
                width=width, height=height, 
                timestep=timestep, camera_id=camera_id))
    return cam_infos, flame_info

def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos

def readDynamicNerfInfo(path, test=False):
    path = os.path.join(path, "UNION")
    if not test:
        print("Reading Training Transforms")
        
        # !!
        train_cam_infos, flame_info = readCamerasFromTransforms(path, "transforms_train.json")
        
        print("Reading Validation Transforms")
        val_cam_infos, _ = readCamerasFromTransforms(path, "transforms_val.json")

        nerf_normalization = getNerfppNorm(train_cam_infos)

        scene_info = SceneInfo(train_cameras=train_cam_infos,
                                val_cameras=val_cam_infos,
                                nerf_normalization=nerf_normalization,
                                flame_info=flame_info)
        
    else:
        print("Reading Test Transforms")
        # test_cam_infos, flame_info = readCamerasFromTransforms(path, "transforms_test.json")
        # 记得改过来
        test_cam_infos, flame_info = readCamerasFromTransforms(path, "transforms_val.json")
        
        nerf_normalization = getNerfppNorm(test_cam_infos)

        scene_info = SceneInfo(train_cameras=[],
                                val_cameras=[],
                                test_cameras=test_cam_infos,
                                nerf_normalization=nerf_normalization,
                                flame_info=flame_info)
    return scene_info

sceneLoadTypeCallbacks = {
    "DynamicNerf" : readDynamicNerfInfo,
}
