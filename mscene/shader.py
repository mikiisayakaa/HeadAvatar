import torch
import numpy as np
from torch import nn
from utils import image_utils as iu
import os
from flame_model.flame import FLAMETex

class DefaultShader():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.texture_height = args.texture_height
        self.texture_width = args.texture_width
        self.envLight = nn.Parameter(torch.ones((16, 32, 1), dtype=torch.float32, device=self.device))
        self.light_activation = nn.Softplus()
        # self.light_activation = nn.Identity()
        self.activation = nn.Identity()
        
        # optim
        self.texture_optimizer = None
        self.light_optimizer = torch.optim.Adam([self.envLight], lr=args.light_lr, eps=1e-15)
    
    def get_light(self):
        return self.light_activation(self.envLight)
    
    def training_update(self, do_update=True):
        if not do_update:
            return
        if self.texture_optimizer is not None:
            self.texture_optimizer.step()
            self.texture_optimizer.zero_grad()
        self.light_optimizer.step()
        self.light_optimizer.zero_grad()
        
    def eval(self):
        pass
        
    def get_albedo(self):
        raise NotImplementedError("get_albedo not implemented")
    
    def get_roughness(self):
        raise NotImplementedError("get_roughness not implemented")
    
    def load(self, path):
        raise NotImplementedError("load not implemented")
    
    def save(self, model_data):
        raise NotImplementedError("save not implemented")
    
class TextureShader(DefaultShader):
    def __init__(self, args, device):
        super().__init__(args, device)

        self.albedo = nn.Parameter(torch.zeros((self.texture_height, self.texture_width, 3), dtype=torch.float32, device=self.device).unsqueeze(0))
        self.roughness = nn.Parameter((torch.ones((self.texture_height, self.texture_width, 1), dtype=torch.float32, device=self.device) * 0.5).unsqueeze(0))
        self.activation = nn.Sigmoid()
        
        # optim
        l_t = []
        l_t.append({'params': [self.albedo], 'lr': args.albedo_lr, "name": "albedo"})
        l_t.append({'params': [self.roughness], 'lr': args.roughness_lr, "name": "roughness"})
        self.texture_optimizer = torch.optim.Adam(l_t, lr=0, eps=1e-15)
        
    def get_albedo(self):
        return self.activation(self.albedo)
    
    def get_roughness(self):
        return self.activation(self.roughness)
    
    def eval(self):
        self.albedo.requires_grad = False
        self.roughness.requires_grad = False
        self.light.requires_grad = False
    
    def load(self, path):
        model_data = torch.load(path)
        self.albedo = nn.Parameter(model_data["albedo"].to(self.device)).unsqueeze(0)
        self.roughness = nn.Parameter(model_data["roughness"].to(self.device)).unsqueeze(0)
        self.light = nn.Parameter(model_data["envLight"].to(self.device))
        
    def save(self, model_data):
        model_data["albedo"] = self.albedo.squeeze(0).detach().cpu()
        model_data["roughness"] = self.roughness.squeeze(0).detach().cpu()
        model_data["envLight"] = self.envLight.detach().cpu()
        return model_data
    
class FixedTexShader(DefaultShader):
    def __init__(self, args, model_data, device):
        super().__init__(args, device)

        self.albedo = model_data["albedo"].to(self.device).unsqueeze(0)
        self.roughness = model_data["roughness"].to(self.device).unsqueeze(0)
        self.envLight = model_data["envLight"].to(self.device)
        self.activation = nn.Sigmoid()
        
        # optim
        self.texture_optimizer = None
        
    def get_albedo(self):
        return self.activation(self.albedo)
    
    def get_roughness(self):
        return self.activation(self.roughness)
    
    def training_update(self, do_update=True):
        pass
    
    def load(self, path):
        pass
        
    def save(self, model_data):
        pass
    
class FLAMETexShader(DefaultShader):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.tex_coeffs = nn.Parameter(torch.zeros((100), dtype=torch.float32, device=self.device))
        
        self.texture_optimizer = torch.optim.Adam([self.tex_coeffs], lr=args.shader_lr, eps=1e-15)
        
        self.flametex = FLAMETex(device)
        
    def get_albedo(self):
        tex = self.flametex(self.tex_coeffs.unsqueeze(0)).permute(0,2,3,1) / 255.0
        return tex
    
    def get_roughness(self):
        return torch.zeros((1, self.texture_height, self.texture_width, 1), dtype=torch.float32, device=self.device)
    
    def load(self, path):
        pass
    
    def save(self, path):
        pass
        
def get_shader(args, device):
    # return FLAMETexShader(args, device)
    shader = TextureShader(args, device)
    
    return shader

def export_texture_vis(shader, path):
    albedo = shader.get_albedo()
    roughness = shader.get_roughness()
    envLight = shader.get_light().unsqueeze(0)
    iu.export_img(os.path.join(path, "albedo.png"), iu.image2numpy(albedo))
    iu.export_img(os.path.join(path, "roughness.png"), iu.image2numpy(roughness))
    iu.export_img(os.path.join(path, "envmap.png"), iu.image2numpy(torch.clamp(envLight, 0, 1)))
            