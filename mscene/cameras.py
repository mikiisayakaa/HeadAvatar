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

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image
import os
from scipy import ndimage

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg, image_width, image, image_height, image_path,
                 image_name, mask_path, semantic_path, uid, camera_id, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 timestep=None, device='cpu'):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_name = image_name
        self.timestep = timestep
        self.semantic_path = semantic_path
        self.camera_id = camera_id
        self.device=device

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        focal = 0.5 * image_width / np.tan(0.5 * FoVx)
        self.focalX = focal
        self.focalY = focal
        self.centerX = (image_width - 1) * 0.5
        self.centerY = (image_height - 1) * 0.5

        self.world_view_transform = getWorld2View2(R, T, trans, scale).T
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).T
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = np.linalg.inv(self.world_view_transform)[3, :3]
        
        self.load()
        # self.reinit(1)
        
    def load(self):
        # load image and mask
        image = Image.open(self.image_path)
        image = np.array(image.convert("RGB"))
        self.original_image = np.clip((image.astype(np.float32) / 255), 0.0, 1.0)
        mask = Image.open(self.mask_path)
        mask = np.array(mask.convert("L"))
        self.original_mask = np.clip((mask.astype(np.float32) / 255), 0.0, 1.0)[..., np.newaxis]
        self.original_image *= self.original_mask
        semantic = Image.open(self.semantic_path)
        semantic = np.array(semantic.convert("L")).astype(np.uint8)
        
        # label map
        # 0 : background
        # 1 : skin (including face and scalp)
        # 2 : left_eyebrow
        # 3 : right_eyebrow
        # 4 : left_eye
        # 5 : right_eye
        # 6 : nose
        # 7 : upper_lip
        # 8 : inner_mouth
        # 9 : lower_lip
        # 10 : hair
        # 11 : left_ear
        # 12 : right_ear
        # 13 : glasses
        C = 14
        self.semantic = np.eye(C)[semantic].astype(np.float32)
        self.semantic_train = np.copy(self.semantic)
        
        skin_channels = [1, 2, 3, 6, 10, 13]
        self.semantic_train[:, :, 1] = self.semantic[:, :, skin_channels].sum(axis=2)
        skin_channels.pop(0)
        self.semantic_train[:, :, skin_channels] = 0.0
        
        self.inner_mouth_mask = self.semantic[..., 8:9] == 0
        
        # left_eye_lmks, left_eye_lmk_mask  = self.get_eye_lmks(eye="left")
        # right_eye_lmks, right_eye_lmk_mask = self.get_eye_lmks(eye="right")
        # self.eye_lmks = np.concatenate([left_eye_lmks, right_eye_lmks], axis=0)
        # self.eye_lmks_mask = np.concatenate([left_eye_lmk_mask, right_eye_lmk_mask], axis=0)
    
    def reinit(self, times=1):
        R = self.R.copy()
        T_offset = -0.01
        T = self.T.copy() + np.array([0, 0, T_offset * times - 0.78])
        # for _ in range(times):
        #     R= R @ self.R_offset
        self.world_view_transform = getWorld2View2(R, T, self.trans, self.scale).T
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).T
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = np.linalg.inv(self.world_view_transform)[3, :3] 
        
    def get_eye_lmks(self, eye="left"):
        if eye == "left":
            column = 4
        else:
            column = 5
        semantic_mask = self.semantic[:, :, column].astype(np.int32)
        labeled_mask, num_features = ndimage.label(semantic_mask)
        
        regions = ndimage.find_objects(labeled_mask)
        if len(regions) == 0:
            print("no eyes detected or only one eye detected!")
            return np.zeros((8, 2), dtype=np.float32), np.zeros((8), dtype=np.float32)
            
        areas = []
        for i, sl in enumerate(regions):
            area = np.sum(labeled_mask[sl] == i)
            areas.append(area)

        max_index = np.argsort(areas)[::-1][0]
        region = regions[max_index]
        
        y_min, y_max = region[0].start, region[0].stop
        x_min, x_max = region[1].start, region[1].stop
            
        eyelmks = []
        x1, x2 = x_min, x_max-1
        xs = [x1, x1*0.75+x2*0.25, x1*0.5+x2*0.5, x1*0.25+x2*0.75, x2]
        sample_xs1 = [xs[0], xs[1], xs[2], xs[3], xs[4]]
        sample_xs2 = [xs[3], xs[2], xs[1]]

        for sample in sample_xs1:
            for y in range(y_min, y_max):
                if semantic_mask[y, int(sample)] == 1:
                    eyelmks.append([int(sample), y])
                    break
        for sample in sample_xs2:
            for y in range(y_max, y_min, -1):
                if semantic_mask[y, int(sample)] == 1:
                    eyelmks.append([int(sample), y])
                    break

        if len(eyelmks) != 8 :
            print("eye landmarks less than 8")
            return np.zeros((8, 2), dtype=np.float32), np.zeros((8), dtype=np.float32)
        
        return np.array(eyelmks, dtype=np.float32), np.ones((8), dtype=np.float32)
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep       