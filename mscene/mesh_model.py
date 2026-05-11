# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# IMPORTANT: add if p.grad is None: continue in optimize.py if you train multi frames
from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import from_differential, to_differential
import utils.mesh_utils as mu
from mscene.mesh import HeadMesh
import os
import nvdiffrast.torch as dr
from mscene.shader import get_shader, export_texture_vis, FixedTexShader
from mscene.vertex import get_vertex_module, FlameModule
from mscene.semantic import (
    FixedSemanticModule, LearnableSemanticModule, export_semantic_vis
)



import utils.image_utils as iu

from flame_model.flame import FlameHead
from flame_model.lbs import blend_shapes
             
class MeshModel():
    def __init__(self, args, flame_info, device='cpu'):
        assert len(args.timesteps) == 1
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.args = args
        self.timesteps = args.timesteps
        self.neural = False
        self.vhap = args.vhap
        self.test = False
        self.specular_pbr = args.specular_pbr

        self.V = 5023
        self.F = 10006
        
        model_data = torch.load("/home/yanp/HeadAvatar/output/new_165_base/final/model_data.pt")
        self.mesh = HeadMesh.load_obj("/home/yanp/HeadAvatar/mesh_uv5.obj", device=self.device)
        
        # self.shader = FixedTexShader(args, model_data, device=self.device)
        self.shader = get_shader(args, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        # for flame texture space only
        # self.mesh.vertices = self.flame_module.flame_model.v_template.clone()
        # self.mesh.faces = self.flame_module.flame_model.faces.to(torch.int32)
        # self.mesh.uvs = self.flame_module.flame_model.verts_uvs
        # self.mesh.uv_ids = self.flame_module.flame_model.textures_idx.to(torch.int32)
        # self.mesh.uvs[:, 1] = 1.0 - self.mesh.uvs[:, 1]
        self.vertex_module = get_vertex_module(args, self.mesh.vertices.clone(), self.mesh.faces.clone(), self.flame_module.static_offset, device=self.device) 
        self.semantic_module = LearnableSemanticModule(args, self.mesh, device=self.device)
        
        self.laplacian = laplacian_uniform(self.mesh.vertices, self.mesh.faces)
        self.adjFaces = mu.getFacePairs(self.laplacian.shape[0], self.mesh.faces)  
        
    def get_vertices(self, t):
        vertices = self.vertex_module.get_vertices(t)
        
        vertices = self.flame_module.forward(vertices, t)
        return vertices
    
    def training_update(self, do_vertex=True, do_texture=True, do_semantic=True):
        self.vertex_module.training_update(do_update=do_vertex)
        self.shader.training_update(do_update=do_texture)
        self.semantic_module.training_update(do_update=do_semantic)
    
    def get_albedo(self):
        return self.shader.get_albedo()
    
    def get_roughness(self):
        return self.shader.get_roughness()
    
    def get_light(self):
        return self.shader.get_light() 
    
    def get_semantic_tex(self):
        return self.semantic_module.get_semantic_tex()     
        
    def save(self, path):
        path = os.path.join(path, "final")
        model_data = {}
        if not os.path.exists(path):
            os.makedirs(path)
        model_data = self.vertex_module.save(model_data)
        model_data = self.shader.save(model_data)
        model_data = self.flame_module.save(model_data)
        model_data = self.semantic_module.save(model_data)
        
        model_data["faces"] = self.mesh.faces.detach().cpu()
        model_data["uvs"] = self.mesh.uvs.detach().cpu()
        model_data["uv_ids"] = self.mesh.uv_ids.detach().cpu()
        torch.save(model_data, os.path.join(path, "model_data.pt"))
        
        export_texture_vis(self.shader, path)
        export_semantic_vis(self.semantic_module, path)
        
    def save_obj(self, path):
        vertices = self.get_vertices(self.timesteps[0])
        mesh = self.mesh.with_vertices(vertices)
        mesh.save_obj(os.path.join(path, "mesh.obj"))

        
class MeshModelBatch(MeshModel):
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.training_args = args
        self.vhap = args.vhap
        self.test = False
        self.neural = args.neural
        self.specular_pbr = args.specular_pbr
        self.differential = True
        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006

        self.timesteps = args.timesteps
        self.model_data = torch.load(os.path.join(path, "model_data.pt"))

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        self.shader = get_shader(args, device=self.device)
        # self.shader = FixedTexShader(args, self.model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        self.vertex_module = get_vertex_module(args, self.mesh.vertices, self.mesh.faces, self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
        
        self.laplacian = laplacian_uniform(self.mesh.vertices, self.mesh.faces)
        self.adjFaces = mu.getFacePairs(self.laplacian.shape[0], self.mesh.faces) 
    
    def save(self, path):
        path = os.path.join(path, "final")
        model_data = {}
        if not os.path.exists(path):
            os.makedirs(path)
        # save offset from base mesh
        v0 = self.mesh.vertices
        expressions = []
        for t in self.timesteps:
            vertices = self.vertex_module.get_vertices(t)
            expressions.append(vertices - v0)
        
        model_data["vertices"] = self.mesh.vertices.detach().cpu()
        model_data["expressions"] = torch.stack(expressions).detach().cpu()
        model_data = self.shader.save(model_data)
        model_data = self.flame_module.save(model_data)
        model_data = self.semantic_module.save(model_data)
        
        # get augmented expr params
        new_flame_basis = self.flame_module.flame_augmentation(vertices, tolerance=1e-6)
        if new_flame_basis is None:
            new_flame_basis = self.flame_module.flame_model.shapedirs[:, :, 300:] 
        
        model_data["faces"] = self.mesh.faces.detach().cpu()
        model_data["uvs"] = self.mesh.uvs.detach().cpu()
        model_data["uv_ids"] = self.mesh.uv_ids.detach().cpu()
        model_data["flame_expr_basis"] = new_flame_basis.detach().cpu()
        torch.save(model_data, os.path.join(path, "model_data.pt"))
        
        export_texture_vis(self.shader, path)
        export_semantic_vis(self.semantic_module, path)
        
class MeshModelBatchTex(MeshModel):
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.training_args = args
        self.neural = args.neural
        self.differential = True
        self.specular_pbr = True
        self.test = False

        path = os.path.join(args.load_path, "final", "model_data.pt")
        self.model_data = torch.load(path)
        
        self.V = 5023
        self.F = 10006

        self.timesteps = args.timesteps

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        # add fixed light
        self.shader = get_shader(args, device=self.device)
        # self.shader.envLight = self.model_data["envLight"].to(self.device)
        
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        
        # get expression added vertices
        vertices = self.mesh.vertices.unsqueeze(0) + self.model_data["expressions"].to(self.device)
        self.vertex_module = get_vertex_module(args, vertices, self.mesh.faces, self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
    
    def save_all_data(self, path):
        path = os.path.join(path, "final")
        if not os.path.exists(path):
            os.makedirs(path)
        
        # only add texture data
        model_data = self.shader.save(self.model_data)
        torch.save(model_data, os.path.join(path, "model_data.pt"))
        export_texture_vis(self.shader, path)   
        
    
class MeshModelTest(MeshModel):
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.training_args = args
        self.vhap = args.vhap
        self.test = False
        self.neural = args.neural
        self.specular_pbr = args.specular_pbr
        self.differential = True
        
        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006

        self.timesteps = args.timesteps
        self.model_data = torch.load(os.path.join(path, "model_data.pt"))

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        
        self.shader = FixedTexShader(args, self.model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        self.vertex_module = get_vertex_module(args, self.mesh.vertices, self.mesh.faces, self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
        
        self.laplacian = laplacian_uniform(self.mesh.vertices, self.mesh.faces)
        self.adjFaces = mu.getFacePairs(self.laplacian.shape[0], self.mesh.faces) 
        
        # self.flame_expr_basis = self.model_data["flame_expr_basis"].to(self.device)
        # self.aug_dim = self.flame_expr_basis.shape[2] - 100 
        
        
class MeshModelRelight:
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.args = args
        self.timesteps = args.timesteps
        self.neural = True
        self.test = True
        self.specular_pbr = True

        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006

        self.model_data = torch.load(os.path.join(path, "model_data.pt"))

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        self.shader = FixedTexShader(args, self.model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        self.vertex_module = get_vertex_module(args, self.mesh.vertices.clone(), self.mesh.faces.clone(), self.flame_module.static_offset, device=self.device) 
        self.semantic_module = LearnableSemanticModule(args, self.mesh, device=self.device)
        
        from PIL import Image
        self.envmap = Image.open(args.envmap_path).convert("RGB")
        self.envmap = self.envmap.resize((1024, 512), Image.BICUBIC)
        self.envmap = np.array(self.envmap).astype(np.float32) / 255.0
        self.envmap = torch.from_numpy(self.envmap).to(self.device)
        self.envmap = self.envmap * 1.0
        self.shader.envLight = self.envmap
        
    def get_vertices(self, t):
        # vertices = self.vertex_module.get_vertices(t)
        
        # vertices = self.flame_module.forward(vertices, t)
        vertices = self.mesh.vertices + self.model_data["expressions"][0].to(self.device)
        vertices = self.flame_module.forward(vertices, t)
        return vertices 
    
    def get_albedo(self):
        return self.shader.get_albedo()
    
    def get_roughness(self):
        return self.shader.get_roughness()
    
    def get_light(self):
        return self.shader.get_light() 
    
    def get_semantic_tex(self):
        return self.semantic_module.get_semantic_tex()   
    
class MeshModelTexEdit:
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.args = args
        self.timesteps = args.timesteps
        self.neural = True
        self.test = True
        self.specular_pbr = True
        
        self.target_color = torch.tensor(args.target_color).to(self.device)
        self.target_semantic = args.target_semantic

        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006

        self.model_data = torch.load(os.path.join(path, "model_data.pt"))

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        self.shader = FixedTexShader(args, self.model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        self.vertex_module = get_vertex_module(args, self.mesh.vertices.clone(), self.mesh.faces.clone(), self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
        
        self.semantic_mask = torch.zeros_like(self.get_semantic_tex()[..., 0:1])
        for semantic_label in self.target_semantic:
            self.semantic_mask += self.get_semantic_tex()[..., semantic_label].unsqueeze(-1)* 0.8
        radius = 11
        coords = torch.arange(-radius, radius+1, device=self.device)
        kernel = torch.exp(-(coords**2) / (2 * radius * radius))
        kernel = kernel / kernel.sum()
        kernel2d = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
        blurred = F.conv2d(self.semantic_mask.permute(0, 3, 1, 2), kernel2d, padding=radius)
        self.semantic_mask = blurred.permute(0, 2, 3, 1)
        iu.export_img("semantic_mask.png", iu.image2numpy(self.semantic_mask.squeeze(0)))
        self.pure_color = self.target_color.view(1, 1, 1, 3).repeat(1, self.texture_height, self.texture_width, 1)
        
        
        
    def get_vertices(self, t):
        # vertices = self.vertex_module.get_vertices(t)
        
        # vertices = self.flame_module.forward(vertices, t)
        vertices = self.mesh.vertices + self.model_data["expressions"][0].to(self.device)
        vertices = self.flame_module.forward(vertices, t)
        return vertices 
    
    def get_albedo(self):
        albedo = self.shader.get_albedo()
        albedo = albedo + self.pure_color * self.semantic_mask
        return albedo
    
    def get_roughness(self):
        return self.shader.get_roughness()
    
    def get_light(self):
        return self.shader.get_light() 
    
    def get_semantic_tex(self):
        return self.semantic_module.get_semantic_tex() 
    
    
class MeshModelExprEdit:
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.args = args
        self.timesteps = args.timesteps
        self.neural = True
        self.test = True
        self.specular_pbr = True

        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006

        self.model_data = torch.load(os.path.join(path, "model_data.pt"))

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        self.shader = FixedTexShader(args, self.model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        self.vertex_module = get_vertex_module(args, self.mesh.vertices.clone(), self.mesh.faces.clone(), self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
    
        self.vertex_semantic = get_vertex_semantic_from_tex(self.semantic_module.get_semantic_tex(), self.mesh)
        fix_labels = [0, 8, 10, 11, 12]
        self.vertex_mask = torch.ones_like(self.vertex_semantic, dtype=torch.float32)
        for label in fix_labels:
            self.vertex_mask[self.vertex_semantic == label] = 0.0
        
        self.exprs = self.model_data["expressions"].to(self.device).clone()
        self.picked_exprs = []
        for i in range(50):
            expr = self.exprs[i]
            expr = expr * self.vertex_mask.unsqueeze(-1)
            expr_dist = torch.norm(expr).item()
            self.picked_exprs.append((expr_dist, expr))
        self.picked_exprs.sort(key=lambda x: x[0], reverse=True)
        self.exprs = [item[1] for item in self.picked_exprs[0:20]]
        # self.exprs.append(torch.zeros((self.V, 3), dtype=torch.float32, device=self.device))
            
            
    def perpare_expr_vertices(self, idx):
        self.expr_vertices = self.mesh.vertices + self.exprs[idx].to(self.device) * self.vertex_mask.unsqueeze(-1)
        
        
    def get_vertices(self, t):
        vertices = self.expr_vertices
        vertices = self.flame_module.forward(vertices, t)
        return vertices 
    
    def get_albedo(self):
        return self.shader.get_albedo()
    
    def get_roughness(self):
        return self.shader.get_roughness()
    
    def get_light(self):
        return self.shader.get_light() 
    
    def get_semantic_tex(self):
        return self.semantic_module.get_semantic_tex() 
    
class MeshModelExprTransfer:
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.args = args
        self.timesteps = args.timesteps
        self.neural = True
        self.test = True
        self.specular_pbr = True

        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006
        
        self.target_model_path = args.another_load_path

        self.model_data = torch.load(os.path.join(path, "model_data.pt"))
        self.target_model_data = torch.load(os.path.join(self.target_model_path, "final", "model_data.pt"))

        self.mesh = HeadMesh(self.target_model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        self.shader = FixedTexShader(args, self.target_model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        # get target flame path
        for folder in os.listdir(args.target_path):
            if folder != "UNION" and os.path.isdir(os.path.join(args.target_path, folder)):
                target_flame_path = os.path.join(args.target_path, folder, "flame_param", "00000.npz")
                target_flame_data = np.load(target_flame_path)["shape"]
                self.flame_module.shape_params = torch.from_numpy(target_flame_data).unsqueeze(0).to(self.device).float()
                break
        
        self.vertex_module = get_vertex_module(args, self.mesh.vertices.clone(), self.mesh.faces.clone(), self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
    
        self.vertex_semantic = get_vertex_semantic_from_tex(self.semantic_module.get_semantic_tex(), self.mesh)
        fix_labels = [0, 8, 10, 11, 12]
        self.vertex_mask = torch.ones_like(self.vertex_semantic, dtype=torch.float32)
        for label in fix_labels:
            self.vertex_mask[self.vertex_semantic == label] = 0.0
        
        
    def get_vertices(self, t):
        vertices = self.mesh.vertices + self.model_data["expressions"][0].to(self.device)
        vertices = self.flame_module.forward(vertices, t)
        return vertices 
    
    def get_albedo(self):
        return self.shader.get_albedo()
    
    def get_roughness(self):
        return self.shader.get_roughness()
    
    def get_light(self):
        return self.shader.get_light() 
    
    def get_semantic_tex(self):
        return self.semantic_module.get_semantic_tex() 
    
class MeshModelIdentityTransfer:
    def __init__(self, args, flame_info, device='cpu'):
        self.texture_height = args.texture_width
        self.texture_width = args.texture_height
        self.device = device
        self.args = args
        self.timesteps = args.timesteps
        self.neural = True
        self.test = True
        self.specular_pbr = True

        path = os.path.join(args.load_path, "final")
        if not os.path.exists(path):
            raise RuntimeError("Load path does not exist: " + path)
        self.V = 5023
        self.F = 10006
        
        self.target_model_path = args.another_load_path

        self.model_data = torch.load(os.path.join(path, "model_data.pt"))
        self.target_model_data = torch.load(os.path.join(self.target_model_path, "final", "model_data.pt"))

        self.mesh = HeadMesh(self.model_data["vertices"].squeeze(0).to(self.device),
                             self.model_data["faces"].to(self.device),
                             self.model_data["uvs"].to(self.device),
                             self.model_data["uv_ids"].to(self.device),
                             device=self.device)
        
        self.shader = FixedTexShader(args, self.target_model_data, device=self.device)
        self.flame_module = FlameModule(args, device=self.device)
        self.flame_module.load_flame(flame_info)
        # get target flame path
        # for folder in os.listdir(args.target_path):
        #     if folder != "UNION" and os.path.isdir(os.path.join(args.target_path, folder)):
        #         target_flame_path = os.path.join(args.target_path, folder, "flame_param", "00000.npz")
        #         target_flame_data = np.load(target_flame_path)["shape"]
        #         self.flame_module.shape_params = torch.from_numpy(target_flame_data).unsqueeze(0).to(self.device).float()
        #         break
        
        self.vertex_module = get_vertex_module(args, self.mesh.vertices.clone(), self.mesh.faces.clone(), self.flame_module.static_offset, device=self.device) 
        self.semantic_module = FixedSemanticModule(args, self.model_data, device=self.device)
    
        self.vertex_semantic = get_vertex_semantic_from_tex(self.semantic_module.get_semantic_tex(), self.mesh)
        fix_labels = [0, 8, 10, 11, 12]
        self.vertex_mask = torch.ones_like(self.vertex_semantic, dtype=torch.float32)
        for label in fix_labels:
            self.vertex_mask[self.vertex_semantic == label] = 0.0
        
        
    def get_vertices(self, t):
        vertices = self.mesh.vertices + self.model_data["expressions"][0].to(self.device)
        vertices = self.flame_module.forward(vertices, t)
        return vertices 
    
    def get_albedo(self):
        return self.shader.get_albedo()
    
    def get_roughness(self):
        return self.shader.get_roughness()
    
    def get_light(self):
        return self.shader.get_light() 
    
    def get_semantic_tex(self):
        return self.semantic_module.get_semantic_tex() 
        
           

def get_mask_from_uv(uvs, image_width=1024, image_height=1024):
    # input uvs should be [N, 3, 2]
    z = torch.zeros_like(uvs[..., 0:1])
    ones = torch.ones_like(uvs[..., 0:1])
    F = uvs.shape[0]
    uv_verts = torch.cat((uvs * 2 - 1, z, ones), dim=-1).reshape(F * 3, 4)
    glctx = dr.RasterizeCudaContext(device=uvs.device)
    
    uv_faces = torch.arange(F * 3, dtype=torch.int32, device=uvs.device).reshape(F, 3)
    rast, _ = dr.rasterize(glctx, uv_verts[None, ...], uv_faces, resolution=[image_height, image_width])
    mask = (rast[..., 3] > 0).squeeze(0)
    
    out_mask = torch.zeros((image_height, image_width, 3), dtype=torch.float32, device=uvs.device)
    out_mask[mask] = 1.0
    
    return out_mask  

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

def get_vertex_semantic_from_tex(semantic_tex, mesh):
    device = semantic_tex.device
    v_semantic = torch.zeros((mesh.vertices.shape[0]), dtype=torch.int64, device=device)
    
    for idx, face in enumerate(mesh.faces):
        for i in range(3):
            vid = face[i].item()
            uv_id = mesh.uv_ids[idx, i].item()
            uv = mesh.uvs[uv_id]
            v_semantic[vid] = torch.argmax(semantic_tex[:, int(uv[1] * (semantic_tex.shape[1] - 1)),
                                                        int(uv[0] * (semantic_tex.shape[2] - 1))])
    
    # semantic_render,_ = get_mask_from_semantic(v_semantic, torch.ones((mesh.vertices.shape[0]), dtype=torch.float32, device=device), 
    #                                               mesh.faces, mesh.uvs, mesh.uv_ids)
    # iu.export_img("semantic_before.png", iu.image2numpy(iu.get_semantic_image(semantic_tex)))
    # iu.export_img("semantic_after.png", iu.image2numpy(iu.get_semantic_image(semantic_render)))
    
    return v_semantic

    
    