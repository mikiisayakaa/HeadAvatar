import torch
from utils import mesh_utils as mu
from largesteps.geometry import laplacian_uniform
from largesteps.parameterize import to_differential, from_differential

class MeshPart:
    def __init__(self, vslice, fslice):
        self.vslice = vslice
        self.fslice = fslice
        self.vlen = vslice.stop - vslice.start
        self.flen = fslice.stop - fslice.start
        
    def update(self, new_voffset, new_foffset):
        self.vslice = slice(new_voffset, new_voffset + self.vlen)
        self.fslice = slice(new_foffset, new_foffset + self.flen)
        
    def clone(self):
        return MeshPart(self.vslice, self.fslice)
    
def getVattr(vattr, part : MeshPart):
    return vattr[part.vslice]

def getFace(faces, part : MeshPart):
    return faces[part.fslice] - part.vslice.start

def getVattrBatch(vattr, part : MeshPart):
    return vattr[:, part.vslice]

def getFaceBatch(faces, part : MeshPart):
    return faces[:, part.fslice] - part.vslice.start


class Mesh:
    def __init__(self, vertices, faces, uvs=None, uv_ids=None, device='cpu'):
        
        self.vertices = vertices
        self.faces = faces
        self.device = device
        self.uvs = uvs
        self.uv_ids = uv_ids
        
    def clone(self):
        return Mesh(self.vertices.clone(), self.faces.clone(),
                    self.uvs.clone(), self.uv_ids.clone(), device=self.device)
    
    def with_vertices(self, vertices):
        return Mesh(vertices, self.faces, self.uvs, self.uv_ids, device=self.device)       
        
        
    def save_obj(self, path):
        
        vertices = self.vertices.detach().cpu().numpy()
        faces = self.faces.cpu().numpy()
        if self.uvs is not None:
            uvs = self.uvs.cpu().numpy()
            uv_ids = self.uv_ids.cpu().numpy()

        with open(path, 'w') as f:
            for v in vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            
            if self.uvs is not None:
                for uv in uvs:
                    f.write(f'vt {uv[0]} {uv[1]}\n')
            
            if self.uvs is not None:
                for idx, face in enumerate(faces):
                    f.write(f'f {face[0]+1}/{uv_ids[idx][0]+1} {face[1]+1}/{uv_ids[idx][1]+1} {face[2]+1}/{uv_ids[idx][2]+1}\n')
            else:
                for idx, face in enumerate(faces):
                    f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    
    @classmethod
    def load_obj(cls, path, device, correct_uv=True):
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
                    
        vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(faces, dtype=torch.int32, device=device)
        uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
        if correct_uv:
            uvs[:, 1] = 1 - uvs[:, 1]
        uv_ids = torch.tensor(uv_ids, dtype=torch.int32, device=device)
        # we split uvs into discrete triangles for simplicity
        fuvs = uvs[uv_ids.long()]
        uvs = fuvs.reshape(-1, 2)
        uv_ids = torch.arange(0, 3 * faces.shape[0], dtype=torch.int32, device=device).reshape(-1, 3)
        
        return cls(vertices, faces, uvs, uv_ids, device) 
    
class HeadMesh(Mesh):
    def __init__(self, vertices, faces, uvs=None, uv_ids=None, device='cpu'):
        super().__init__(vertices, faces, uvs, uv_ids, device)
        # self.skinPart = MeshPart(slice(0, 3931), slice(0, 7800))
        # self.eye1Part = MeshPart(slice(3931, 4477), slice(7800, 8888))
        # self.eye2Part = MeshPart(slice(4477, 5023), slice(8888, 9976))
        self.skinPart = MeshPart(slice(0, 3931), slice(0, 7830))
        self.eye1Part = MeshPart(slice(3931, 4477), slice(7830, 8918))
        self.eye2Part = MeshPart(slice(4477, 5023), slice(8918, 10006))
        # self.skinPart = MeshPart(slice(0, 7460), slice(0, 14846))
        # self.eye1Part =  MeshPart(slice(7460, 8006), slice(14846, 15934))
        # self.eye2Part =  MeshPart(slice(8006, 8552), slice(15934, 17022))
        # subdiv
        # self.skinPart = MeshPart(slice(0, 15691), slice(0, 31320))
        # self.eye1Part = MeshPart(slice(15691, 16237), slice(31320, 32408))
        # self.eye2Part = MeshPart(slice(16237, 16783), slice(32408, 33496))
        
    def with_vertices(self, vertices):
        return HeadMesh(vertices, self.faces, self.uvs, self.uv_ids, device=self.device) 
        
    def rearrange(self):
        edges = mu.createEdges(self.faces)
        new_faces, vidmaps, fidmaps = mu.split_mesh(edges, self.faces)
        new_vertices = mu.split_mesh_vattributes(vidmaps, self.vertices)
        new_uv_ids = mu.split_mesh_fattributes(fidmaps, self.uv_ids)
        self.faces = mu.combine_mesh(new_faces)
        self.vertices = torch.cat(new_vertices, dim=0)
        self.uv_ids = torch.cat(new_uv_ids, dim=0)
        
        # self.meshParts = {
        #     "skin": MeshPart(slice(0, 3931), slice(0, 7800)),
        #     "eye1": MeshPart(slice(3931, 4477), slice(7800, 8888)),
        #     "eye2": MeshPart(slice(4477, 5023), slice(8888, 9976)),
        # }
        # create MeshParts
        # parts = []
        # voffset = 0
        # foffset = 0
        # for i in range(len(new_faces)):
        #     part = MeshPart(slice(voffset, voffset + new_vertices[i].shape[0]),
        #                     slice(foffset, foffset + new_faces[i].shape[0]))
        #     parts.append(part)
        #     voffset += new_vertices[i].shape[0]
        #     foffset += new_faces[i].shape[0]
            
        
    
            
        
        
            
        