import os
import torch
from torch.utils.data import DataLoader
from mesh_renderer_torch import render
from mscene import Scene, MeshModelExprTransfer
from argparse import ArgumentParser
from arguments import MeshModelExprTransferParams
from utils import image_utils as iu
    
def test(mp, device):
    torch.cuda.set_device(device)
    
    scene = Scene(mp, device)
    scene.load_cameras(mp.timesteps)
    mesh_model = MeshModelExprTransfer(mp, scene.flame_info, device)
    
    os.makedirs("expr_transfer_results", exist_ok=True)

    for timestep in mp.timesteps:
        print(f"Testing timestep {timestep}")
        val_loader = DataLoader(scene.getValCameras(), 1, shuffle=False, collate_fn=scene.collate_fn)
        

        for cam_batch in val_loader:
            renderer = render(cam_batch, mesh_model, device)
            image = renderer.colors
            image = iu.image2numpy(image)
            iu.export_img(os.path.join("expr_transfer_results", f"{str(timestep).zfill(5)}_{str(cam_batch.camera_id[0]).zfill(2)}.png"), image)     
                

            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpuID", type=int, default=0)
    mp = MeshModelExprTransferParams(parser)

    args = parser.parse_args()
    
    
    mesh_params = mp.extract(args)
    device = "cuda:" + str(args.gpuID)
    
    test(mesh_params, device)
    