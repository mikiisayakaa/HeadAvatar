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

from tqdm import tqdm
from mscene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(id, cam_info):
    orig_w, orig_h = cam_info.width, cam_info.height

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image_width=orig_w, image_height=orig_h,
                  bg=cam_info.bg, 
                  image=cam_info.image, 
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name,
                  mask_path=cam_info.mask_path,
                  semantic_path=cam_info.semantic_path,
                  uid=id, 
                  camera_id=cam_info.camera_id,
                  timestep=cam_info.timestep)
    
def cameraList_from_camInfos(cam_infos, timesteps, ids):
    cams = []
    for id, c in enumerate(cam_infos):
        if c.timestep not in timesteps:
            continue
        if c.camera_id in ids:
            cams.append(loadCam(id, c))
            
    return cams

# def cameraList_from_camInfos(cam_infos, args):
#     train_cams = []
#     val_cams = []

#     for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
#         # we only need a single frame
#         if c.timestep not in args.timesteps:
#             continue
#         if c.camera_id in args.train_ids:
#             train_cams.append(loadCam(id, c))
#         elif c.camera_id in args.val_ids:
#             val_cams.append(loadCam(id, c))

#     return train_cams, val_cams

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
