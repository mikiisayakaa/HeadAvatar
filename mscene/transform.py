import torch
from torch import nn

class Transform(nn.Module):
    def __init__(self, batch_size=1, device='cpu'):
        super(Transform, self).__init__()
        self.batch_size = batch_size
        self.device = device
        # axis-angle
        self.rotation = torch.zeros((self.batch_size, 3), dtype=torch.float32, device=device)
        self.translation = torch.zeros((self.batch_size, 3), dtype=torch.float32, device=device)
        self.scale = torch.ones([self.batch_size, 1], dtype=torch.float32, device=device)
            
    def training_setup(self, lr, from_center=False, fix=[]):
        l_t = []
        self.lr = lr
        self.fix = fix
        self.from_center = from_center
        if "rotation" not in fix:
            self.rotation = nn.Parameter(self.rotation, requires_grad=True)
            l_t.append({'params' : self.rotation, 'lr' : lr})
        if "translation" not in fix:
            self.translation = nn.Parameter(self.translation, requires_grad=True)
            l_t.append({'params' : self.translation, 'lr' : lr})
        if "scale" not in fix:
            self.scale = nn.Parameter(self.scale, requires_grad=True)
            l_t.append({'params' : self.scale, 'lr' : lr})
            
        return l_t
    
    def get_transform_matrix(self):
        theta = torch.norm(self.rotation, dim=-1, keepdim=True) + 1e-8  # [B, 1]
        axis = self.rotation / theta  # [B, 3]
        scale = self.scale.unsqueeze(-1)

        # 构造 K 矩阵 [B, 3, 3]
        zero = torch.zeros_like(axis[:, 0])
        K = torch.stack([
            zero, -axis[:, 2], axis[:, 1],
            axis[:, 2], zero, -axis[:, 0],
            -axis[:, 1], axis[:, 0], zero
        ], dim=-1).reshape(-1, 3, 3)

        I = torch.eye(3, device=self.device).unsqueeze(0)  # [1, 3, 3]
        K2 = torch.bmm(K, K)

        # Rodrigues公式计算旋转矩阵 [B, 3, 3]
        R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * K2
        R = R * scale

        # 构建 [B, 4, 4] 仿射矩阵
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)  # [B, 4, 4]
        T[:, 0:3, 0:3] = R
        T[:, 0:3, 3] = self.translation

        return T
        
    def clone(self):
        new_t = Transform(device=self.device)
        new_t.rotation = self.rotation.clone()
        new_t.translation = self.translation.clone()
        new_t.scale = self.scale.clone()
        
        new_t.fix = self.fix
        new_t.lr = self.lr
        new_t.from_center = self.from_center
        
        return new_t
    
    def clone_detach(self):
        new_t = Transform(device=self.device)
        new_t.rotation = self.rotation.clone().detach()
        new_t.rotation.requires_grad_(False)
        new_t.translation = self.translation.clone().detach()
        new_t.translation.requires_grad_(False)
        new_t.scale = self.scale.clone().detach()
        new_t.scale.requires_grad_(False)
        
        new_t.fix = self.fix
        new_t.lr = self.lr
        new_t.from_center = self.from_center
        
        return new_t
          
    def forward(self, vertices):
        if len(vertices.shape) == 2:
            v = vertices[None, ...]
        else:
            v = vertices
        B, N, _ = v.shape  # [B, N, 3]
        ones = torch.ones(B, N, 1, device=self.device)
        homo = torch.cat([v, ones], dim=-1)  # [B, N, 4]

        if self.from_center:
            center = torch.mean(v, dim=1, keepdim=True)  # [B, 1, 3]
            homo[..., 0:3] -= center

        T = self.get_transform_matrix()  # [B, 4, 4]
        homo = homo.transpose(1, 2)  # [B, 4, N]

        new_verts = torch.bmm(T, homo)  # [B, 4, N]
        new_verts = new_verts.transpose(1, 2)[..., 0:3]  # [B, N, 3]

        if self.from_center:
            new_verts += center
            
        if len(vertices.shape) == 2:
            new_verts = new_verts.squeeze(0)

        return new_verts
        
        # ones = torch.ones_like(vertices[..., 0:1], device=self.device)
        # homo = torch.cat([vertices, ones], dim=-1)
        
        # # construct transformation matrix
        # trans = torch.eye(4, device=self.device)
        # rot = self.rotation / (torch.norm(self.rotation, dim=1, keepdim=True) + 1e-8)
        # rot0 = rot[0]
        # rot2 = torch.cross(rot[0], rot[1])
        # rot2 = rot2 / (torch.norm(rot2) + 1e-8)
        # rot1 = torch.cross(rot2, rot0)
        # rot1 = rot1 / (torch.norm(rot1) + 1e-8)
        # trans[0, 0:3] = rot0
        # trans[1, 0:3] = rot1
        # trans[2, 0:3] = rot2
        # trans[0:3, 0:3] *= self.scale
        # trans[0:3, 3] = self.translation
        
        # if self.from_center:
        #     # move to origin -> transform -> move back
        #     center = torch.mean(vertices, dim=0)
        #     homo[..., 0:3] -= center
        
        # shape = homo.shape
        # new_verts = torch.matmul(trans, homo.reshape(-1, 4).T)
        # new_verts = new_verts.T.reshape(shape)
        
        # new_verts = new_verts[..., 0:3] / new_verts[..., 3:4]
        
        # if self.from_center:
        #     new_verts += center
            
        # return new_verts 
    
    def __str__(self):
        return (
            "Transform:-----------------\n" + \
            f"rotation:{self.rotation}\n" + \
            f"translation:{self.translation}\n" + \
            f"scale:{self.scale}\n"
        )        
