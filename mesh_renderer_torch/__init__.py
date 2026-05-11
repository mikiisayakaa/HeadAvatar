import torch
from torch import nn
import nvdiffrast.torch as dr
from mscene.shading import bruteforce_specular_shader2, bruteforce_specular_shader2_batch
from utils import mesh_utils as mu
from typing import NamedTuple

PI = 3.1415926
from utils.timer_utils import CudaTimer

class MeshRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    cam_center: torch.Tensor

def render(viewpoint, mesh_model, device):
    renderer = MeshRendererTorch(device)
    renderer(viewpoint, mesh_model)
    return renderer

class MeshRendererTorch(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device
        
    def forward(self, viewpoint, mesh_model):
        timer = CudaTimer()
        # with timer.record("raster_setup:"):
        self.raster_settings = MeshRasterizationSettings(
            image_height=int(viewpoint.image_height),
            image_width=int(viewpoint.image_width),
            viewmatrix=viewpoint.world_view_transform,
            projmatrix=viewpoint.full_proj_transform,
            cam_center=viewpoint.camera_center,
        )
        B = self.raster_settings.viewmatrix.shape[0]
        self.batch_size = B
        
        self.specular = torch.ones((B, self.raster_settings.image_height, self.raster_settings.image_width, 3), dtype=torch.float32, device=self.device)
        
        vertices = torch.cat([mesh_model.get_vertices(t)[None, ...] for t in viewpoint.timesteps], dim=0)

        self.vertices = vertices
        self.mesh = mesh_model.mesh
        with timer.record("rasterize and shade:"):
            result = self._render(vertices, self.mesh, mesh_model)
        timer.summary()
        return result
        
    def _render(self, vertices, mesh, mesh_model):
        B = self.batch_size
        ones = torch.ones((B, vertices.shape[1], 1), dtype=torch.float32, device=self.device)
        vpos = torch.cat((vertices, ones), dim=-1)
        v_hom = vpos @ self.raster_settings.projmatrix

        rast, _ = dr.rasterize(self.glctx, v_hom, mesh.faces, resolution=[
            self.raster_settings.image_height,
            self.raster_settings.image_width
        ])
        self.rast = rast
        
        mask = (rast[..., 3:] > 0)
        self.boolmask = mask
        self._mask = mask.view(B, -1)
        
        fnormals = mu.compute_face_normals(vertices, mesh.faces)
        vnormals = mu.compute_vertex_normals(vertices, mesh.faces, fnormals)
        
        normal, _ = dr.interpolate(vnormals, rast, mesh.faces)
        self.normal = self.get_masked_tensor_batch(normal)
        
        pos, _ = dr.interpolate(vertices.contiguous(), rast, mesh.faces)
        self.pos = self.get_masked_tensor_batch(pos)
        self.v = vertices
        cam_center = self.raster_settings.cam_center
        self.view_dir = []
        
        # backface culling
        self.backface_culling = []
        for ib in range(B):
            view_dir = cam_center[ib] - self.pos[ib]
            view_dir = view_dir / torch.linalg.norm(view_dir,dim=-1,keepdim=True)
            self.view_dir.append(view_dir)
            backface_culling = torch.ones([view_dir.shape[0], 1], dtype=torch.float32, device=view_dir.device) 
            self.backface_culling.append(backface_culling)
        
        texc, _ = dr.interpolate(mesh.uvs.unsqueeze(0).repeat(B, 1, 1), rast, mesh.uv_ids)
        self.texc = self.get_masked_tensor_batch(texc)

        
        specular = dr.texture(self.specular, texc, filter_mode='linear')
        self.specular = self.get_masked_tensor_batch(specular)
        albedo = dr.texture(mesh_model.get_albedo().contiguous(), texc, filter_mode='linear')
        self.albedo = self.get_masked_tensor_batch(albedo)
        roughness = dr.texture(mesh_model.get_roughness().contiguous(), texc, filter_mode='linear')
        self.roughness = self.get_masked_tensor_batch(roughness)
        light = mesh_model.get_light()
        
        c2w_rot = self.raster_settings.viewmatrix[:, :3, :3] @ torch.tensor([[
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        
        colors = []
        shadings = []
        roughness = []
        specular = []
        normals = []
        for ib in range(B):
            shading, diffuse_shading = bruteforce_specular_shader2(
                self.normal[ib], self.view_dir[ib],
                self.albedo[ib], self.specular[ib], self.roughness[ib], light.permute(2, 0, 1).repeat(3, 1, 1),
                c2w_rot[ib], enable_specular = mesh_model.specular_pbr
            )

            colors.append(self.get_image(shading, ib))
            shadings.append(self.get_image(diffuse_shading, ib))
            roughness.append(self.get_image(self.roughness[ib], ib))
            normals.append(self.get_image(self.normal[ib], ib))
        
        self.colors = dr.antialias(torch.cat(colors, dim=0), rast, v_hom, mesh.faces, pos_gradient_boost=1)
        self.shading = dr.antialias(torch.cat(shadings, dim=0), rast, v_hom, mesh.faces)
        self.roughness = dr.antialias(torch.cat(roughness, dim=0), rast, v_hom, mesh.faces)
        self.mask = dr.antialias(self.boolmask.float(), rast, v_hom, mesh.faces)
        self.normal = dr.antialias(torch.cat(normals, dim=0), rast, v_hom, mesh.faces)
        
        semantic_tex = mesh_model.get_semantic_tex()
        self.learnable_semantic = dr.texture(semantic_tex.repeat(B, 1, 1, 1), texc, filter_mode='linear')
        self.learnable_semantic = dr.antialias(self.learnable_semantic, rast, v_hom, mesh.faces) 
        
    def _render_test(self, mesh, albedo, roughness, light, camera):
        self.specular = torch.ones((1, 1024, 1024, 3), dtype=torch.float32, device=self.device)
        vertices = mesh.vertices[None, ...]
        ones = torch.ones((1, vertices.shape[1], 1), dtype=torch.float32, device=self.device)
        vpos = torch.cat((vertices, ones), dim=-1)
        v_hom = vpos @ camera.projmatrix
        
        rast, _ = dr.rasterize(self.glctx, v_hom, mesh.faces, resolution=[802, 550])
        
        mask = (rast[..., 3:] > 0)
        self.boolmask = mask
        self._mask = mask.view(1, -1)
        
        fnormals = mu.compute_face_normals(vertices, mesh.faces)
        vnormals = mu.compute_vertex_normals(vertices, mesh.faces, fnormals)
        
        normal, _ = dr.interpolate(vnormals, rast, mesh.faces)
        self.normal = self.get_masked_tensor_batch(normal)
        
        pos, _ = dr.interpolate(vertices.contiguous(), rast, mesh.faces)
        self.pos = self.get_masked_tensor_batch(pos)
        self.v = vertices
        cam_center = camera.cam_center
        self.view_dir = []
        
        # backface culling
        self.backface_culling = []
        view_dir = cam_center - self.pos[0]
        view_dir = view_dir / torch.linalg.norm(view_dir,dim=-1,keepdim=True)
        self.view_dir.append(view_dir)
        backface_culling = torch.ones([view_dir.shape[0], 1], dtype=torch.float32, device=view_dir.device) 
        backface_culling[(view_dir * self.normal[0]).sum(dim=-1) <= 0.0] = 0
        self.backface_culling.append(backface_culling)
        
        texc, _ = dr.interpolate(mesh.uvs.unsqueeze(0), rast, mesh.uv_ids)
        self.texc = texc

        albedo = dr.texture(albedo.unsqueeze(0), texc, filter_mode='linear')
        self.albedo = self.get_masked_tensor_batch(albedo)
        specular = dr.texture(self.specular, texc, filter_mode='linear')
        self.specular = self.get_masked_tensor_batch(specular)
        roughness = dr.texture(roughness.unsqueeze(0), texc, filter_mode='linear')
        self.roughness = self.get_masked_tensor_batch(roughness)
        
        c2w_rot = camera.viewmatrix[:3, :3] @ torch.tensor([[
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]], dtype=torch.float32, device=self.device)
        
        shading, _ = bruteforce_specular_shader2(
            self.normal[0], self.view_dir[0],
            self.albedo[0], self.specular[0], self.roughness[0], light.permute(2, 0, 1).repeat(3, 1, 1),
            c2w_rot, enable_specular = True
        )
            
        colors = self.get_image(shading, 0)

        colors = dr.antialias(colors, rast, v_hom, mesh.faces, pos_gradient_boost=1)

        return colors.squeeze(0)
    
    def get_masked_tensor_batch(self, x):
        C = x.shape[-1]
        batch_size = x.shape[0]
        masked_set = []
        for ib in range(batch_size):
            masked_set.append(x[ib].view(-1,C)[self._mask[ib]])
        # return list of tensor Pi x C
        return masked_set
    
    def get_image(self, x, id):
        # x: list of tensor Pi x C
        C = x.shape[-1]
        B, H, W, _ = self.boolmask.shape
        canvas = torch.zeros([H, W, C], dtype=torch.float32, device=self.device)
        canvas[self.boolmask[id].squeeze(-1).contiguous()] = (x * self.backface_culling[id]).contiguous()
        # canvas[self.boolmask[id].squeeze(-1).contiguous()] = x.contiguous()
        return canvas.view(1, H, W, C)

            
        
        
        
        
        
    
    
            
            
            
            
            
            
            
            