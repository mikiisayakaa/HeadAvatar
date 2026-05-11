import torch
import numpy as np
import os
import cv2
import utils.mesh_utils as mu

class MeshModelView():
    def __init__(self, path, device):
        
        self.device = device

        mesh_path = os.path.join(path, "mesh.obj")
        
        vertices = []
        faces = []
        uvs = []
        uv_ids = []
        
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.strip().split(' ')[1:]])
                elif line.startswith('vt '):
                    uvs.append([float(x) for x in line.strip().split(' ')[1:]])
                elif line.startswith('f '):
                    face = [int(x.split('/')[0]) - 1 for x in line.strip().split(' ')[1:]]
                    uv_id = [int(x.split('/')[1]) - 1 for x in line.strip().split(' ')[1:]]
                    faces.append(face)
                    uv_ids.append(uv_id)
                    
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        self.faces = torch.tensor(faces, dtype=torch.int32, device=device)
        self.uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
        # uvs[:, 1] = 1 - uvs[:, 1]
        self.uv_ids = torch.tensor(uv_ids, dtype=torch.int32, device=device)
        
        tex_path = os.path.join(path, "textures")
        
        def load_tex(filename):
            file_path = os.path.join(tex_path, filename) 
            tex = cv2.imread(file_path)
            if tex.shape[2] == 3:
                tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
            tex_tensor = torch.from_numpy(np.array(tex) / 255).to(torch.float32).to(self.device)
            return tex_tensor
            
        self.albedo = load_tex("albedo.png")
        self.roughness = load_tex("roughness.png")
        self.specular = load_tex("specular.png")
        self.envmap = load_tex("envmap.png").permute(2, 0, 1)
        # import pdb
        # pdb.set_trace()
        # self.envmap.fill_(1)        
        fnormals = mu.compute_face_normals(self.vertices, self.faces)
        self.vnormals = mu.compute_vertex_normals(self.vertices, self.faces, fnormals)
        