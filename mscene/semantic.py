import torch
import numpy as np
import nvdiffrast.torch as dr
import torch.nn.functional as F
from mscene import mesh_model
from utils import image_utils as iu
import os

def read_tensor_from_txt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            num = int(line.strip())
            data.append(num)
    return torch.tensor(data, dtype=torch.int)

class DefaultSemanticModule():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_semantic = 14
        
    def get_semantic_tex(self):
        return self.fixed_semantic * (1 - self.learnable_semantic_mask) + self.learnable_semantic * self.learnable_semantic_mask
    
    def save(self, model_data):
        semantic = self.get_semantic_tex().detach().cpu()
        # convert to one-hot
        semantic_1d = semantic.reshape(-1, self.num_semantic)
        semantic_indices = torch.argmax(semantic_1d, dim=-1)
        semantic_new = torch.zeros_like(semantic_1d)
        semantic_new[torch.arange(semantic_1d.shape[0]), semantic_indices.long()] = 1.0
        semantic = semantic_new.reshape(1, semantic.shape[1], semantic.shape[2], self.num_semantic)
        model_data["semantic"] = semantic.squeeze(0)
        return model_data

class LearnableSemanticModule(DefaultSemanticModule):
    def __init__(self, args, mesh, device):
        super().__init__(args, device)
        
        V = mesh.vertices.shape[0]
        
        self.vertex_semantic =  read_tensor_from_txt("/home/yanp/HeadAvatar/_mapped.txt").to(self.device)
        # self.vertex_semantic = torch.ones_like(self.vertex_semantic)
        
        semantic_mask = (self.vertex_semantic == 1).to(torch.float32)
        
        # fix for eye region
        self.eye1Orbital = read_tensor_from_txt("/home/yanp/HeadAvatar/left_eye_orbital4.txt").to(self.device).long()
        self.eye2Orbital = read_tensor_from_txt("/home/yanp/HeadAvatar/right_eye_orbital4.txt").to(self.device).long()
        eyeOrbital = torch.cat([self.eye1Orbital, self.eye2Orbital], dim=0)
        semantic_mask[eyeOrbital] = 0.0

        self.learnable_semantic, self.learnable_semantic_mask = get_mask_from_semantic(self.vertex_semantic, semantic_mask, mesh.faces, mesh.uvs, mesh.uv_ids)
        self.fixed_semantic = self.learnable_semantic.clone().detach()
        self.learnable_semantic = torch.nn.Parameter(self.learnable_semantic, requires_grad=True)

        l_s = []
        l_s.append({'params': [self.learnable_semantic], 'lr': self.args.semantic_lr, "name": "semantic"})
        self.optimizer = torch.optim.Adam(l_s, lr=0.0, eps=1e-15)
        
    def training_update(self, do_update=True):
        if not do_update:
            return
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
class FixedSemanticModule(DefaultSemanticModule):
    def __init__(self, args, model_data, device):
        super().__init__(args, device)
        
        self.learnable_semantic = model_data["semantic"].to(self.device).unsqueeze(0)
        self.fixed_semantic = self.learnable_semantic.clone().detach()
        self.learnable_semantic_mask = torch.zeros_like(self.learnable_semantic)   
        
    def training_update(self, do_update=True):
        return   
        
        
def get_mask_from_semantic(semantic, mask, faces, uvs, uv_ids, image_width=1024, image_height=1024):
    z = torch.zeros_like(uvs[..., 0:1])
    ones = torch.ones_like(uvs[..., 0:1])
    uv_verts = torch.cat([uvs * 2 - 1, z, ones], dim=-1)
    
    semantic_verts = semantic
    V = semantic_verts.shape[0]
    semantic_onehot = torch.zeros([V, 14], dtype=torch.float32, device=semantic.device)
    semantic_onehot[torch.arange(V, device=semantic.device).long(), semantic_verts.long()] = 1.0
    mask_verts = mask
    glctx = dr.RasterizeCudaContext(device=uvs.device)
    
    rast, _ = dr.rasterize(glctx, uv_verts[None, ...], uv_ids, resolution=[image_height, image_width])
    semantic_tex, _ = dr.interpolate(semantic_onehot[None, ...].to(torch.float32).contiguous(), rast, faces)
    texmask, _ = dr.interpolate(mask_verts[None, ..., None].to(torch.float32).contiguous(), rast, faces)
    semantic_tex = dr.antialias(semantic_tex, rast, uv_verts, uv_ids)
    texmask = dr.antialias(texmask, rast, uv_verts, uv_ids)
    # 膨胀一下，覆盖掉某些缝隙
    semantic_tex = F.max_pool2d(semantic_tex.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1).permute(0, 2, 3, 1)
    texmask = F.max_pool2d(texmask.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1).permute(0, 2, 3, 1)

    return semantic_tex, texmask

def export_semantic_vis(semantic_module, path):
    semantic_tex = iu.get_semantic_image(semantic_module.get_semantic_tex())
    # semantic_tex = semantic_tex.permute(0, 3, 1, 2)
    # semantic_tex = F.interpolate(semantic_tex, size=(1024, 1024), mode='bilinear', align_corners=False)
    # semantic_tex = semantic_tex.permute(0, 2, 3, 1).squeeze(0)
    iu.export_img(os.path.join(path, "semantic.png"), iu.image2numpy(semantic_tex))