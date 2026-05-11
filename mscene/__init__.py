import os
import torch
from typing import List
from arguments import GroupParams
from mscene.dataset_readers import sceneLoadTypeCallbacks
from mscene.mesh_model import MeshModel, MeshModelBatch, MeshModelBatchTex, MeshModelTest, MeshModelRelight, MeshModelTexEdit, MeshModelExprEdit, MeshModelExprTransfer, MeshModelIdentityTransfer
from copy import deepcopy
from mscene.cameras import Camera
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import numpy as np

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int): 
            # camera = deepcopy(self.cameras[idx])
            # camera.load()         
            # return camera
            return self.cameras[idx]
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        elif isinstance(idx, list):
            return CameraDataset([self.cameras[i] for i in idx])
        else:
            raise TypeError("Invalid argument type")
        
def camera_collate_fn(batch):
    class CameraBatch:
        def __init__(self):
            pass
        
    cam_batch = CameraBatch()
    cam_batch.image_width = batch[0].image_width
    cam_batch.image_height = batch[0].image_height
    cam_batch.zfar = batch[0].zfar
    cam_batch.znear = batch[0].znear
    cam_batch.device = batch[0].device
    cam_batch.timesteps = [camera.timestep for camera in batch]
    cam_batch.camera_id = [camera.camera_id for camera in batch]

    def to_torch_and_stack(attr_name):
        setattr(cam_batch, attr_name, torch.stack([torch.from_numpy(getattr(camera, attr_name)).to(cam_batch.device) for camera in batch], dim=0))

    to_torch_and_stack("world_view_transform")
    to_torch_and_stack("full_proj_transform")
    to_torch_and_stack("camera_center")
    to_torch_and_stack("original_image")
    to_torch_and_stack("original_mask")
    to_torch_and_stack("semantic")
    to_torch_and_stack("semantic_train")
    to_torch_and_stack("inner_mouth_mask")
    if hasattr(batch[0], "exp_params"):
        to_torch_and_stack("exp_params")
        to_torch_and_stack("full_pose")
        to_torch_and_stack("shape_params")  
        
    if hasattr(batch[0], "multiframe"):
        cam_batch.multiframe = True      
    
    return cam_batch

class Scene:
    def __init__(self, args : GroupParams, device):

        # load dataset
        assert os.path.exists(args.source_path), "Source path does not exist: {}".format(args.source_path)
        scene_info = sceneLoadTypeCallbacks["DynamicNerf"](args.source_path, args.test)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.scene_info = scene_info
        self.device = device
        
        self.collate_fn = camera_collate_fn
        self.args = args
        
    def load_cameras(self, timesteps):
        self.train_cameras = cameraList_from_camInfos(self.scene_info.train_cameras, timesteps, self.args.train_ids)
        self.val_cameras = cameraList_from_camInfos(self.scene_info.val_cameras, timesteps, self.args.val_ids)
        for camera in self.train_cameras:
            camera.device = self.device
        for camera in self.val_cameras:
            camera.device = self.device
        
        self.flame_info = {}
        for timestep in timesteps:
            self.flame_info[timestep] = self.scene_info.flame_info[timestep]
            
    def load_cameras_test(self, timesteps):
        self.test_cameras_train = cameraList_from_camInfos(self.scene_info.test_cameras, timesteps, self.args.test_train_ids)
        self.test_cameras_val = cameraList_from_camInfos(self.scene_info.test_cameras, timesteps, self.args.test_val_ids)
        for camera in self.test_cameras_train:
            camera.device = self.device
        for camera in self.test_cameras_val:
            camera.device = self.device

        self.flame_info = {}
        for timestep in timesteps:
            self.flame_info[timestep] = self.scene_info.flame_info[timestep]

    def getTrainCameras(self):
        return CameraDataset(self.train_cameras)
    
    def getValCameras(self):
        return CameraDataset(self.val_cameras)

    def getTestTrainCameras(self, timestep):
        cameras = []
        for camera in self.test_cameras_train:
            if camera.timestep == timestep:
                cameras.append(camera)
        return CameraDataset(cameras)
    
    def getTestValCameras(self, timestep):
        cameras = []
        for camera in self.test_cameras_val:
            if camera.timestep == timestep:
                cameras.append(camera)
        return CameraDataset(cameras)