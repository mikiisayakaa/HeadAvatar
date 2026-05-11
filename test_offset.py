import torch
from flame_model.flame import FlameHead
from flame_model.lbs import blend_shapes
from mscene.mesh import HeadMesh
from mesh_renderer_torch import MeshRendererTorch
import os
from utils import image_utils as iu
from collections import deque
import cv2
import argparse

class SimpleCamera:
    def __init__(self):
        self.viewmatrix = torch.eye(4)
        self.cam_center = torch.tensor([0.0, 0.0, 5.0])
        self.projmatrix = torch.eye(4)
        
def semantic_BFS_floodfill(semantic, semantic_mask, tex_mask):
    q = deque()
    # semantic_mask: 0 不可学习，1 可学习
    # tex_mask: 0 未更新，1 已更新
    semantic = semantic.squeeze(0)
    out_semantic = semantic.clone()
    h, w, _ = semantic.shape
    
    for i in range(h):
        for j in range(w):
            if tex_mask[i, j] > 0.5 and semantic_mask[i, j] > 0.5:
                q.append((i, j))
                
    while q:
        x, y = q.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if tex_mask[nx, ny] < 0.5 and semantic_mask[nx, ny] > 0:
                    out_semantic[nx, ny] = semantic[x, y]
                    tex_mask[nx, ny] = 1
                    q.append((nx, ny))
                    
    return out_semantic

def pca_torch(offsets, k):
    # offsets: (N, 5023, 3)
    N = offsets.shape[0]
    
    # flatten
    X = offsets.reshape(N, -1)  # (N, 15069)

    # center
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    # SVD-based PCA
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    # principal components
    components = Vh[:k]  # (k, 15069)

    # projection
    X_pca = Xc @ components.T  # (N, k)

    return X_pca, components, mean          
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    path = f"/home/yanp/HeadAvatar/output/new/new_{args.model}_seq/final/model_data.pt"
    image_name = f"offset_{args.model}.png"
    image_pca_name = f"offset_pca_{args.model}.png"
    # single_path = "/home/yanp/HeadAvatar/output/test_1106_165_base/final/"
    os.makedirs("./offset_test", exist_ok=True)
    timestep = 3
    timestep1 = 35
    data = torch.load(path)
    device = "cuda:1"
    
    flame_model = FlameHead(300, 100, device=device,include_mask=False, add_teeth=False).to(device)
    # visualize semantic
    # semantic = data["semantic"].to(device)
    # tex_mask = torch.tensor(cv2.imread(os.path.join(single_path, "texmask.png"), cv2.IMREAD_GRAYSCALE) / 255.0, dtype=torch.float32).to(device)
    # semantic_mask = torch.tensor(cv2.imread("./semantic_mask.png", cv2.IMREAD_GRAYSCALE) / 255.0, dtype=torch.float32).to(device)
    # iu.export_img("semantic_orig.png", iu.image2numpy(iu.get_semantic_image(semantic)))
    # out_semantic = semantic_BFS_floodfill(semantic, semantic_mask, tex_mask)
    # iu.export_img("semantic_floodfill.png", iu.image2numpy(iu.get_semantic_image(out_semantic[None,...])))
    
    
    # visualize expr offset mesh
    camera = SimpleCamera()
    # view 10
    camera.cam_center = torch.tensor([-0.3163,  0.2317,  1.0484]).to(device)
    camera.viewmatrix = torch.tensor([[ 0.9698, -0.0519,  0.2384,  0.0000],
         [ 0.0068, -0.9710, -0.2391,  0.0000],
         [ 0.2439,  0.2335, -0.9413,  0.0000],
         [ 0.0494, -0.0362,  1.1177,  1.0000]], dtype=torch.float32).to(device)
    camera.projmatrix = torch.tensor([[ 7.2248, -0.2650,  0.2384,  0.2384],
         [ 0.0509, -4.9608, -0.2391, -0.2391],
         [ 1.8168,  1.1929, -0.9414, -0.9413],
         [ 0.3683, -0.1852,  1.1078,  1.1177]], dtype=torch.float32).to(device)
    
    renderer = MeshRendererTorch(device)
    images = []
    activation_img = torch.nn.Sigmoid()
    activation_light = torch.nn.Softplus()
    expressions = []
    dists = []
    for timestep in range(0, 49):
        mesh = HeadMesh(
            data["vertices"].to(device) + data["expressions"][timestep].to(device),
            data["faces"].to(device),
            data["uvs"].to(device),
            data["uv_ids"].to(device),
            device=device
        )
        dist = torch.norm(data["expressions"][timestep]).item()
        if dist < 0.4:
            continue
        dists.append(dist)
        
        expressions.append(data["expressions"][timestep].unsqueeze(0).to(device))
        color = renderer._render_test(mesh, 
                                      activation_img(data["albedo"].to(device)), 
                                      activation_img(data["roughness"].to(device)), 
                                      activation_light(data["envLight"].to(device)),
                                      camera)
        
        images.append(color)
        
    iu.visualize_grid(images, 7, image_name)
    
    print(dists)
    
    # expressions = torch.cat(expressions, dim=0)  # (N, V, 3)
    # k = 10
    # pca_coeff, pca_components, mean = pca_torch(expressions, k=k)
    # print(pca_coeff)
    # images_pca = []
    # for i in range(49):
    #     mesh = HeadMesh(
    #         data["vertices"].to(device) + (pca_coeff[i:i+1] @ pca_components + mean.reshape(1, -1)).reshape(5023, 3),
    #         data["faces"].to(device),
    #         data["uvs"].to(device),
    #         data["uv_ids"].to(device),
    #         device=device
    #     )
        
    #     color = renderer._render_test(mesh, 
    #                                   activation_img(data["albedo"].to(device)), 
    #                                   activation_img(data["roughness"].to(device)), 
    #                                   activation_light(data["envLight"].to(device)),
    #                                   camera)
        
    #     images_pca.append(color)
        
    # iu.visualize_grid(images_pca, 7, image_pca_name)
    # mesh1 = HeadMesh(
    #     data["vertices"].to(device) + data["expressions"][timestep1].to(device),
    #     data["faces"].to(device),
    #     data["uvs"].to(device),
    #     data["uv_ids"].to(device),
    #     device=device
    # )
    # mesh1.save_obj(os.path.join("offset_test", f"{timestep1}.obj"))
    
    # coeff1 = 0.5
    # coeff2 = 0.5
    # v_combined = mesh1.vertices * coeff1 + mesh.vertices * coeff2
    # mesh_combined = mesh1.with_vertices(v_combined)
    # mesh_combined.save_obj(os.path.join("offset_test", f"{timestep}_{coeff1}-{timestep1}_{coeff2}_combined.obj"))
    
    # mesh = mesh.with_vertices(data["vertices"])
    # mesh.save_obj(f"{timestep}_orig.obj")
    
    # mesh = mesh.with_vertices(data["vertices"] + 3 * data["expressions"][timestep])
    # mesh.save_obj(f"{timestep}_offset.obj")
    