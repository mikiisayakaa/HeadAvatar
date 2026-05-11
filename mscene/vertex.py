import torch
import numpy as np
from torch import nn

from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import from_differential, to_differential

from flame_model.flame import FlameHead

class DefaultVertexModule():
    def __init__(self, args, vertices, device):
        self.args = args
        self.device = device
        self.timesteps = args.timesteps
        self.original_verticees = vertices
        
        self.optimizer = None
    
    def get_index(self, t):
        return self.timesteps.index(t)
    
    def get_vertices(self, t):
        raise NotImplementedError("get_vertices not implemented")
    
    def training_update(self, do_update=True):
        if not do_update:
            return
        
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
    def save(self, model_data):
        raise NotImplementedError("save not implemented")
            
    
class VertexOffsetModule(DefaultVertexModule):
    def __init__(self, args, vertices, device):
        super().__init__(args, vertices, device)

        if vertices.shape[0] == 1: 
            self.vertices = nn.Parameter(vertices.clone()).unsqueeze(0).repeat(len(self.timesteps), 1, 1)
        else:
            self.vertices = nn.Parameter(vertices.clone()[None, ...])
        self.optimizer = torch.optim.Adam([self.vertices],  lr=args.position_lr, eps=1e-15)
        
    def get_vertices(self, t):
        idx = self.get_index(t)
        return self.vertices[idx, ...]
    
    def save(self, model_data):
        model_data["vertices"] = self.vertices.detach().cpu()
        return model_data

class StaticOffsetModule(DefaultVertexModule):
    def __init__(self, args, vertices, device):
        super().__init__(args, vertices, device)
        
        self.vertices = vertices.squeeze()
        
    def get_vertices(self, t):
        return self.vertices
    
    def training_update(self, do_update=True):
        pass
    
    def save(self, model_data):
        model_data["vertices"] = self.vertices.unsqueeze(0).detach().cpu()
        return model_data
    
class DiffVertexModule(DefaultVertexModule):
    def __init__(self, args, vertices, faces, device):
        super().__init__(args, vertices, device)
       
        self.laplacian = compute_matrix(vertices, faces, lambda_=19)
        self.diffVertices = to_differential(self.laplacian, vertices)
        self.diffVertices = nn.Parameter(self.diffVertices.unsqueeze(0).repeat(len(self.timesteps), 1, 1), requires_grad = True)
        self.optimizer = AdamUniform([self.diffVertices], lr=self.args.position_lr)
    
    def get_vertices(self, t):
        idx = self.get_index(t)
        vertices = from_differential(self.laplacian, self.diffVertices[idx, ...])
        return vertices
    
    def save(self, model_data):
        vert_list = []
        for i in range(len(self.timesteps)):
            vertices = from_differential(self.laplacian, self.diffVertices[i, ...])
            vert_list.append(vertices.unsqueeze(0))
        model_data["vertices"] = torch.cat(vert_list, dim=0).detach().cpu()
        return model_data            
    
class FlameModule():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.timesteps = args.timesteps
        
    def load_flame(self, flame_info):
        self.shape_params = None
        self.static_offset = None
        self.expr_params = []
        self.rotation = []
        self.neck_pose = []
        self.jaw_pose = []
        self.eyes_pose = []
        self.translation = []
        for timestep, flame_path in flame_info.items():
            flame_data = np.load(flame_path, allow_pickle=True)
            if self.shape_params is None:
                self.shape_params = torch.from_numpy(flame_data["shape"]).to(self.device).unsqueeze(0)
            if self.static_offset is None and "static_offset" in flame_data:
                self.static_offset = torch.from_numpy(flame_data["static_offset"][:, :5023, :]).to(self.device)
            self.expr_params.append(torch.from_numpy(flame_data["expr"]).to(self.device))
            self.rotation.append(torch.from_numpy(flame_data["rotation"]).to(self.device))
            self.neck_pose.append(torch.from_numpy(flame_data["neck_pose"]).to(self.device))
            self.jaw_pose.append(torch.from_numpy(flame_data["jaw_pose"]).to(self.device))
            self.eyes_pose.append(torch.from_numpy(flame_data["eyes_pose"]).to(self.device))
            self.translation.append(torch.from_numpy(flame_data["translation"]).to(self.device))
            
        self.expr_params = torch.cat(self.expr_params, dim=0)
        self.rotation = torch.cat(self.rotation, dim=0)
        self.neck_pose = torch.cat(self.neck_pose, dim=0)
        self.jaw_pose = torch.cat(self.jaw_pose, dim=0)
        self.eyes_pose = torch.cat(self.eyes_pose, dim=0)
        self.translation = torch.cat(self.translation, dim=0)

        self.flame_model = FlameHead(300, 100, device=self.device, include_mask=False, add_teeth=False).to(self.device)
    
    def forward(self, vertices, t):
        idx = self.timesteps.index(t)
        vertices = self.flame_model.forward(
            vertices,
            self.shape_params,
            self.expr_params[idx:idx+1],
            self.rotation[idx:idx+1],
            self.neck_pose[idx:idx+1],
            self.jaw_pose[idx:idx+1],
            self.eyes_pose[idx:idx+1],
            self.translation[idx:idx+1],
            return_landmarks=False,
            return_verts_cano=False
        ).squeeze(0)
        
        return vertices
    
    def flame_augmentation(self, new_shape_components, tolerance=1e-6):
        flame_components = self.flame_model.shapedirs[:, :, 300:]
        flame_dim = flame_components.shape[2]
        flame_components = flame_components.reshape(-1, flame_dim)
        new_components = new_shape_components.reshape(5023 * 3, -1)
        projector = torch.matmul(flame_components, flame_components.t()) #[V*3, V*3]
        residual = new_components - torch.matmul(projector, new_components)
        
        R_norm = torch.norm(residual, dim=0)
        if torch.all(R_norm < tolerance):
            print("No new shape components added.")
            return None
        
        # SVD
        U, sigma, Vt = torch.svd(residual)
        # 可以做一下根据特征值保留
        
        new_basis = torch.cat([flame_components, U], dim=1)
        new_basis = new_basis.reshape(5023, 3, -1)
        return new_basis
        
    def save(self, model_data):
        flame_joints = torch.cat((self.rotation, self.neck_pose, self.jaw_pose, self.eyes_pose, self.translation), dim=1)
        model_data["flame_joints"] = flame_joints.detach().cpu()
        model_data["shape_params"] = self.shape_params.detach().cpu()
        model_data["expr_params"] = self.expr_params.detach().cpu()
        model_data["static_offset"] = self.static_offset.detach().cpu()
        return model_data
        
def get_vertex_module(args, vertices, faces, static_offset, device):
    if args.differential:
        print("Using differential vertex module")
        vmodule = DiffVertexModule(args, vertices, faces, device)
    elif args.vhap:
        print("Using static offset vertex module")
        vmodule = StaticOffsetModule(args, vertices + static_offset, device)
    else:
        print("Using vertex offset module")
        vmodule = VertexOffsetModule(args, vertices, device)
    return vmodule
    
        
        
        


