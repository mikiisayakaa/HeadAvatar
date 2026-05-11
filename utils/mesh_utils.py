import torch
import numpy as np


def faceEquations(face_verts : torch.Tensor):
    '''
        Calculate face equations with shape [F, 4]
        One equation: ax + by + cz + d = 0 -> [a, b, c, d]
        Input: [F, 3, 3]
    '''
    e1 = face_verts[:, 1] - face_verts[:, 0]
    e2 = face_verts[:, 2] - face_verts[:, 0]
    nrm = torch.cross(e1, e2, dim=1)
    d = -(nrm[:, 0] * face_verts[:, 0, 0] + nrm[:, 1] * face_verts[:, 0, 1] + nrm[:, 2] * face_verts[:, 0, 2])
    
    return torch.cat((nrm, d.unsqueeze(0)), dim=1)

def QEM(faces : torch.Tensor, face_verts : torch.Tensor, V : int):
    '''
        Calculate vertex QEM matrices for evaluating decimation
    '''
    faceEqs = faceEquations(face_verts)
    faceQEM = torch.bmm(faceEqs.unsqueeze(2), faceEqs.unsqueeze(1))
    
    vertQEM = torch.zeros((V, 4, 4), dtype=face_verts.dtype, device=face_verts.device)
    
    for i in range(3):
        vertQEM.scatter_add_(
            dim=0,  
            index=faces[:, i].view(-1, 1, 1).expand(-1, 4, 4), 
            src=faceQEM
        )
        
    return vertQEM

def createEdges(faces : torch.Tensor):
    '''
        Create a unique edge index list, small index first
    '''
    # sort face index first
    f_sort, _ = torch.sort(faces, dim=1)
    F = faces.shape[0]
    
    # key: v1 * F + v2
    keys = torch.cat([f_sort[:, 0] * F + f_sort[:, 1],
                      f_sort[:, 0] * F + f_sort[:, 2],
                      f_sort[:, 1] * F + f_sort[:, 2]], dim=0)
    unique_keys = torch.unique(keys)
    
    return torch.stack([unique_keys // F, unique_keys % F], dim=-1)

def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges

def QEMError(edgeQEM : torch.Tensor, homo : torch.Tensor):
    '''
        Error = xT @ QEM @ x
    '''
    return torch.bmm(torch.bmm(homo.squeeze(1), edgeQEM), homo.squeeze(2)).squeeze()

def solveDestinationAndError(vertQEM : torch.Tensor, vertices : torch.tensor, edges : torch.Tensor):
    '''
        Core evaluation function for QEM decimation
    '''
    # construct equations for all edges
    edgeQEM = vertQEM[edges[:, 0]] + vertQEM[edges[:, 1]]
    A = edgeQEM[:, 0:3, 0:3]
    b = -edgeQEM[:, 0:3, 3]
    
    # use determinant to exclude equations without solution
    detA = torch.linalg.det(A)
    zerodet = detA < 1e-5
    id_solvable = torch.nonzero(~zerodet).squeeze()
    id_unsolvable = torch.nonzero(zerodet).squeeze()
    
    # pass to process solvable equations
    A_solvable = A[id_solvable]
    b_solvable = b[id_solvable]
    
    coords_solvable = torch.linalg.solve(A_solvable, b_solvable)
    homo_solvable = torch.cat([coords_solvable,
                               torch.ones((coords_solvable.shape[0], 1), 
                                          dtype=vertQEM.dtype, device=vertQEM.device)], dim=-1)
    error_solvable = QEMError(edgeQEM[id_solvable], homo_solvable)
    
    # pass to process unsolvable equations, check several points on the edge and choose the best
    edge_unsolvable = edges[id_unsolvable]
    v0 = vertices[edge_unsolvable[:, 0]]
    v1 = vertices[edge_unsolvable[:, 1]]
    checknum = 10
    candidates = []
    
    for i in range(checknum + 1):
        a0 = 1 - i / checknum
        a1 = i / checknum
        candidates.append(a0 * v0 + a1 * v1)
    
    candidates = torch.stack(candidates, dim=1)
    candidates = torch.cat([candidates, 
                            torch.ones((candidates.shape[0], candidates.shape[1], 1), 
                                       dtype=vertQEM.dtype, device=vertQEM.device)], dim=-1)
    homo_unsolvable = candidates.reshape((-1, 3))
    eQEM_unsolvable = edgeQEM[id_unsolvable][:, None, ...].expand(-1, checknum + 1, 4, 4)
    eQEM_unsolvable = eQEM_unsolvable.reshape(-1, 4, 4)
    error_unsolvable = QEMError(eQEM_unsolvable, homo_unsolvable)
    error_unsolvable = error_unsolvable.reshape(-1, checknum + 1)
    error_unsolvable, minidx = torch.min(error_unsolvable, dim=1)
    coords_unsolvable = candidates[:, minidx, 0:3]
    
    # combine both parts of output
    out_coords = torch.zeros((edges.shape[0], 3), dtype=vertQEM.dtype, device=vertQEM.device)
    out_errs = torch.zeros((edges.shape[0]), dtype=vertQEM.dtype, device=vertQEM.device)
    out_coords[id_solvable] = coords_solvable
    out_coords[id_unsolvable] = coords_unsolvable
    out_errs[id_solvable] = error_solvable
    out_errs[id_unsolvable] = error_unsolvable
    
    return out_coords, out_errs

def getVertexFaceMap(face : torch.Tensor, V : int):
    '''
        Get a list mapping vertices to adjacent faces.
    '''
    adjFaces = []
    for i in range(V):
        adjFaces.append([])
    
    for idx, f in enumerate(face):
        adjFaces[f[0]].append(idx)
        adjFaces[f[1]].append(idx)
        adjFaces[f[2]].append(idx)
        
    return adjFaces
        
def getVertexEdgeMap(edges : torch.Tensor, V : int):
    '''
        Get a list mapping vertices to adjacent edges.
    '''
    adjEdges = []
    for i in range(V):
        adjEdges.append([])
        
    for idx, e in enumerate(edges):
        adjEdges[e[0]].append(idx)
        adjEdges[e[1]].append(idx)
        
    return adjEdges 

def getVertexVertexMap(edges : torch.Tensor, V : int):
    adjVerts = []
    for i in range(V):
        adjVerts.append([])
        
    for idx, e in enumerate(edges):
        adjVerts[e[0]].append(e[1].item())
        adjVerts[e[1]].append(e[0].item())
        
    return adjVerts   
    
        
def decimate(vertices : torch.Tensor, 
             faces : torch.Tensor, 
             vuvs : torch.Tensor, 
             target_num : int):
    V = vertices.shape[0]
    F = faces.shape[0]
    device = vertices.device
    if V <= target_num:
        return vertices, faces, vuvs
    if target_num <= 0:
        raise RuntimeError("decimate(): target_num <= 0")
    iterations = V - target_num
    
    # int64 for indices
    fs = faces.long()
    
    face_verts = torch.stack([vertices[fs[:, 0]], vertices[fs[:, 1]], vertices[fs[:, 2]]], dim=-1)
    vertQEM = QEM(fs, face_verts, V)
    
    edges = createEdges(fs)
    E = edges.shape[0]
    v2f = getVertexFaceMap(fs, V)
    v2e = getVertexEdgeMap(edges, V)
    outcoords, errors = solveDestinationAndError(vertQEM, vertices, edges)
    
    _, indices = torch.sort(errors)
    
    # we use three sets to note the wrong vertex/face/edge data
    v_empty = set()
    f_empty = set()
    e_empty = set()
     
    for i in range(iterations):
        # remove one vertex in one iter
        cur_eidx = indices[0]
        cur_vout = outcoords[cur_eidx]
        cur_e = edges[cur_eidx]
        ring1verts = set()
        
        # remove all relevant geometry elements
        v_empty.add(cur_e[0].item())
        v_empty.add(cur_e[1].item())
        
        # pick an empty vertex as new vertex
        vnew = v_empty.pop()
        vertices[vnew] = cur_vout
        
        # rearrange topology
        for v in cur_e:
            v0 = v.item()
            v1 = cur_e[1].item() if v0 == cur_e[0] else cur_e[0].item()
            for e in v2e[v0]:
                ve = edges[e, 0].item() if edges[e, 1] == v0 else edges[e, 1].item()
                if ve != v1:
                    # normal edge, create a new edge between v0 and vnew
                    edgepair = [vnew, v0] if vnew < v0 else [v0, vnew]
                    v2e[v0].remove(e)
                    v2e[vnew].append(e)
                    edges[e] = torch.tensor(edgepair, dtype=edges.dtype, device=device)
                else:
                    # edge between v0 and v1, delete it
                    v2e[v0].remove(e)
                    v2e[v1].remove(e)
                    e_empty.add(e)
            for f in v2f[v0]:
                vf1 = faces[f, 0].item() if faces[f, 0] != v0 else faces[f, 2].item()
                vf2 = faces[f, 1].item() if faces[f, 1] != v0 else faces[f, 2].item()
    '''
        TODO: Complete this
    '''
                
def getFacePairs(V: int, faces : torch.Tensor):    
    # map edge to face
    facemap = {}
    facepairs = []
    for idx, f in enumerate(faces):
        for i in range(3):
            v1 = f[i].item()
            v2 = f[(i + 1) % 3].item()
            if v1 > v2:
                v1, v2 = v2, v1
            key = v1 * V + v2
            if key not in facemap:
                facemap[key] = idx
            else:
                facepairs.append([idx, facemap[key]])      
    adj_faces = torch.tensor(facepairs, dtype=torch.int64, device=faces.device)
    return adj_faces
                
def laplacian_smooth(vertices : torch.Tensor, faces : torch.Tensor, iterations : int, factor : float):
    device = vertices.device
    fs = faces.long()
    edges = createEdges(fs)
    adjVerts = getVertexVertexMap(edges, vertices.shape[0])
    
    for iter in range(iterations):
        for idx, adj in enumerate(adjVerts):
            indices = torch.tensor(adj, dtype=fs.dtype, device=device)
            avgpoint = torch.sum(vertices[indices], dim=0) / indices.shape[0]
            vertices[idx] += (avgpoint - vertices[idx]) * factor
            
def compute_laplacian_uniform(V : int, faces : torch.Tensor):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    edges_packed = createEdges(faces)

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))
    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=faces.device)
    A = torch.sparse.FloatTensor(idx.to(torch.int64), ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx.to(torch.int64), val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=faces.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=faces.device)
    L -= torch.sparse.FloatTensor(idx.to(torch.int64), ones, (V, V))

    return L

def compute_face_normals(vertices : torch.Tensor, faces : torch.Tensor):
    # Compute the face normals
    a = vertices[:, faces[:, 0].long(), :]
    b = vertices[:, faces[:, 1].long(), :]
    c = vertices[:, faces[:, 2].long(), :]
    face_normals = torch.cross(b - a, c - a, dim=-1)
    norm = torch.norm(face_normals, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    return face_normals / norm

def compute_vertex_normals(vertices : torch.Tensor, faces : torch.Tensor, face_normals : torch.Tensor):
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals = vertex_normals.index_add(1, faces[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(1, faces[:, 1], face_normals)
    vertex_normals = vertex_normals.index_add(1, faces[:, 2], face_normals)
    vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 
    return vertex_normals

def compute_triangle_angles(verts, faces):
    edges = []
    edges.append(verts[..., faces[..., 1], :] - verts[..., faces[..., 0], :])
    edges.append(verts[..., faces[..., 2], :] - verts[..., faces[..., 0], :])
    edges.append(verts[..., faces[..., 2], :] - verts[..., faces[..., 1], :])
    
    for i in range(3):
        edges[i] = torch.nn.functional.normalize(edges[i], dim=-1)
    
    angles = []
    angles.append(torch.sum(edges[0]*edges[1], dim=-1, keepdim=True))
    angles.append(torch.sum(edges[1]*edges[2], dim=-1, keepdim=True))
    angles.append(torch.sum(-edges[2]*edges[0], dim=-1, keepdim=True))
    angles = torch.cat(angles, dim=-1)
    
    return angles
    

def create_edgemap(edges : torch.Tensor):
    E = edges.shape[0]
    keys = (edges[:, 0] * E + edges[:, 1]).tolist()
    edgemap = {}
    
    for eidx, key in enumerate(keys):
        edgemap[key] = eidx
        
    return edgemap

def getEidxFromEdgemap(edgemap : dict, face : torch.Tensor, i : int):
    v1 = face[i].item()
    v2 = face[(i + 1) % 3].item()
    if v1 > v2:
        v1, v2 = v2, v1
    key = v1 * len(edgemap) + v2
    eidx = edgemap[key]
    return eidx

def edge_to_face(edges : torch.Tensor, faces : torch.Tensor):
    E = edges.shape[0]
    edge2face = -torch.ones((E, 2), dtype=torch.int64, device=faces.device)
    edgemap = create_edgemap(edges)
    
    for fidx, face in enumerate(faces):
        for i in range(3):
            eidx = getEidxFromEdgemap(edgemap, face, i)
            if edge2face[eidx, 0] == -1:
                edge2face[eidx, 0] = fidx
            else:
                edge2face[eidx, 1] = fidx
    
    boundary_edges = []            
    for e in range(E):
        if edge2face[e, 1] == -1:
            boundary_edges.append(e)
    boundary_edges = torch.tensor(boundary_edges, dtype=torch.int32, device=faces.device)
                
    return edge2face, boundary_edges
    

def split_mesh(edges : torch.Tensor, faces : torch.Tensor):
    '''
        Split mesh into disjoint sub-meshes
        
        Returns 1-d vidmaps and fidmaps, records the old index of face and vertices
    '''
    E = edges.shape[0]
    edgemap = create_edgemap(edges)
    edge2face, _ = edge_to_face(edges, faces)
    
                
    # BFS
    face_processed = torch.zeros(faces.shape[0], dtype=bool, device=faces.device)
    vblocks = []
    meshes = []
    queue = []
    blockIdx = -1
    
    fidmaps = []

    while not (len(queue) == 0 and torch.all(face_processed)):
        if len(queue) == 0:
            vblocks.append(set())
            meshes.append(torch.zeros((0, 3), dtype=torch.int64, device=faces.device))
            fidmaps.append([])
            blockIdx += 1
            for idx in range(len(face_processed)):
                if face_processed[idx] == 0:
                    queue.append(idx)
                    face_processed[idx] = 1
                    break

        fidx = queue.pop()
        face = faces[fidx]
        vblocks[blockIdx].add(faces[fidx, 0].item())
        vblocks[blockIdx].add(faces[fidx, 1].item())
        vblocks[blockIdx].add(faces[fidx, 2].item())
        fidmaps[blockIdx].append(fidx)
        meshes[blockIdx] = torch.cat([meshes[blockIdx], faces[fidx][None, :]], dim=0)
    
        for i in range(3):
            eidx = getEidxFromEdgemap(edgemap, face, i)
            efaces = edge2face[eidx]
            f1 = efaces[0].item()
            f2 = efaces[1].item()
            if f1 >= 0 and face_processed[f1] == 0:
                queue.append(f1)
                face_processed[f1] = 1
            if f2 >= 0 and face_processed[f2] == 0:
                queue.append(f2)
                face_processed[f2] = 1
            
    # remap vertex ids
    vidmaps = []
    new_meshes = []
    
    for idx in range(len(vblocks)):
        vlist = torch.tensor(list(vblocks[idx]), dtype=torch.int64, device=faces.device)
        vlist, _ = torch.sort(vlist)
        newids = torch.arange(len(vlist), device=faces.device)
        vidmaps.append(vlist)
        fidmaps[idx] = torch.tensor(fidmaps[idx], dtype=torch.int64, device=faces.device) 
        
        map_tensor = torch.empty((vlist.max() + 1), dtype=torch.int64, device=faces.device)
        map_tensor[vlist] = newids
        new_f = []
        for i in range(3):
            old_vids = meshes[idx][:, i]
            new_vids = map_tensor[old_vids]
            new_f.append(new_vids[:, None])
        new_f = torch.cat(new_f, dim=1)
        new_meshes.append(new_f.to(torch.int32))
        
    sorted_lists = sorted(zip(new_meshes, vidmaps, fidmaps), key=lambda x: x[0].shape[0], reverse=True)
    return map(list, zip(*sorted_lists))

def split_mesh_vattributes(vidmaps : list[torch.Tensor], attr : torch.Tensor):
    attr_list = []
    for idx in range(len(vidmaps)):
        attr_list.append(attr[vidmaps[idx]])
        
    return attr_list

def split_mesh_fattributes(fidmaps : list[torch.Tensor], attr : torch.Tensor):
    attr_list = []
    for idx in range(len(fidmaps)):
        attr_list.append(attr[fidmaps[idx]])
    
    return attr_list

def combine_mesh(face_lists : list[torch.Tensor]):
    vsum = 0
    faces = torch.zeros((0, 3), dtype=torch.int32, device=face_lists[0].device)
    for idx in range(len(face_lists)):
        V = torch.max(face_lists[idx]) + 1
        cur_faces = face_lists[idx] + vsum
        faces = torch.cat([faces, cur_faces], dim=0)
        vsum += V
        
    return faces


def combine_mesh_attributes(attr_list : list[torch.Tensor]):
    return torch.cat(attr_list, dim=0)

def barycentric2D(A, B, C, P):
    a = (-(P[0] - B[0]) * (C[1] - B[1]) + (P[1] - B[1]) * (C[0] - B[0])) / \
        (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0]))
    b = (-(P[0] - C[0]) * (A[1] - C[1]) + (P[1] - C[1]) * (A[0] - C[0])) / \
        (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0]))
    c = 1.0 - a - b
    return np.array([a, b, c], dtype=np.float32)

def barycentric2D_batch(A, B, C, P):
    a = (-(P[:, 0] - B[0]) * (C[1] - B[1]) + (P[:, 1] - B[1]) * (C[0] - B[0])) / \
        (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0]))
    b = (-(P[:, 0] - C[0]) * (A[1] - C[1]) + (P[:, 1] - C[1]) * (A[0] - C[0])) / \
        (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0]))
    c = 1.0 - a - b
    return torch.stack([a, b, c], dim=1)

def barycentric_projection(A, B, C, P):
    # return the barycentric of projection point and distance to plane
    normal = np.cross(B - A, C - A)
    normal = normal / np.linalg.norm(normal)
    A_, B_, C_ = normal

    D = -np.dot(normal, A)

    # calculate projection P
    t = -(np.dot(normal, P) + D) 
    d = abs(t)
    P_proj = P + t * normal

    # calculate barycentric coordinate
    v0 = B - A
    v1 = C - A
    v2 = P_proj - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01 

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return [u, v, w], d

def loop_subdivision(vertices, faces):
    '''
        Returns : new vertices, new faces, mapping from new vertices idxs to old 2 edge vertices idxs
    '''       
    V = vertices.shape[0]
    F = faces.shape[0]
    
    edges = find_edges(faces)
    
    E = edges.shape[0]
    edgemap = create_edgemap(edges)
    edge2face, boundary_edges = edge_to_face(edges, faces)
    vert2vert = []
    boundary_verts = edges[boundary_edges].flatten().unique()
    
    for i in range(V):
        vert2vert.append([])
        
    for i in range(E):
        e = edges[i]
        v1 = e[0].item()
        v2 = e[1].item()
        if v1 not in vert2vert[v2]:
            vert2vert[v1].append(v2)
        if v2 not in vert2vert[v1]:
            vert2vert[v2].append(v1)
            
    # create new vertices
    temp_verts = [0, 0, 0, 0, 0, 0]
    new_verts = torch.zeros((E, 3), dtype=vertices.dtype, device=vertices.device)
    new_faces = torch.zeros((F * 4, 3), dtype=faces.dtype, device=faces.device)
    for i in range(F):
        f = faces[i]
        for j in range(3):
            e = getEidxFromEdgemap(edgemap, f, j)
            v1 = f[j].item()
            v2 = f[(j + 1) % 3].item()
            temp_verts[j * 2] = v1
            temp_verts[j * 2 + 1] = e + V
        
        for j in range(3):
            new_faces[i * 4 + j, 0] = temp_verts[j * 2 + 1]
            new_faces[i * 4 + j, 1] = temp_verts[(j * 2 + 2) % 6]
            new_faces[i * 4 + j, 2] = temp_verts[(j * 2 + 3) % 6]
            new_faces[i * 4 + 3, j] = temp_verts[j * 2 + 1]
            
    # calculate positions of new vertices
    for i in range(E):
        e = edges[i]
        v1 = e[0].item()
        v2 = e[1].item()
        if v1 in boundary_verts and v2 in boundary_verts:
            new_verts[i] = (vertices[v1] + vertices[v2]) * 1 / 2
            continue
        f1 = faces[edge2face[i, 0].item()]
        f2 = faces[edge2face[i, 1].item()]
        for j in range(3):
            if f1[j] != v1 and f1[j] != v2:
                v3 = f1[j]
                break
        for j in range(3):
            if f2[j] != v1 and f2[j] != v2:
                v4 = f2[j]
                break
        new_verts[i] = (vertices[v1] + vertices[v2]) * 3 / 8 + \
                       (vertices[v3] + vertices[v4]) * 1 / 8
                       
    # calculate positions of old vertices
    old_verts_update = vertices.clone()
    for i in range(V):
        degree = len(vert2vert[i])
        # fix boundary vertices
        if i in boundary_verts:
            continue
        if degree <= 3:
            u = 3 / 16
        else:
            u = 3 / (8 * degree)
        old_verts_update[i] *= (1 - degree * u)
        for j in range(degree):
            old_verts_update[i] += u * vertices[vert2vert[i][j]]
            
    new_verts = torch.cat((old_verts_update, new_verts), dim=0)
    new_vert_map = torch.zeros((E, 3), dtype=faces.dtype, device=faces.device)
    for i in range(E):
        e = edges[i]
        v1 = e[0].item()
        v2 = e[1].item()
        new_vert_map[i, 0] = i + V
        new_vert_map[i, 1] = v1
        new_vert_map[i, 2] = v2
        
    return new_verts, new_faces, new_vert_map
    
                    
            
    
                
                    

            
            
            
            
            
            
        
    
    
    
    