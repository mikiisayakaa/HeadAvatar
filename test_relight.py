import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mesh_renderer_torch import render
from mscene import Scene, MeshModelRelight
from utils.general_utils import safe_state
from lpipsPyTorch import lpips
from argparse import ArgumentParser
from arguments import MeshModelRelightParams
from utils import image_utils as iu
    
def test(mp, device):
    torch.cuda.set_device(device)
    
    scene = Scene(mp, device)
    scene.load_cameras(mp.timesteps)
    mesh_model = MeshModelRelight(mp, scene.flame_info, device)
    
    first_iter = 0 
    
    os.makedirs("relighting_results", exist_ok=True)

    for timestep in mp.timesteps:
        print(f"Testing timestep {timestep}")
        train_loader = DataLoader(scene.getTrainCameras(), 1, shuffle=False, collate_fn=scene.collate_fn)
        val_loader = DataLoader(scene.getValCameras(), 1, shuffle=False, collate_fn=scene.collate_fn)

        for cam_batch in val_loader:
            renderer = render(cam_batch, mesh_model, device)
            image = renderer.colors
            image = iu.image2numpy(image)
            iu.export_img(os.path.join("relighting_results", f"{str(timestep).zfill(5)}_{str(cam_batch.camera_id[0]).zfill(2)}.png"), image)
            light = mesh_model.shader.get_light()
            light = iu.image2numpy(light)
            iu.export_img("test_light.png", light)        

            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpuID", type=int, default=0)
    mp = MeshModelRelightParams(parser)

    args = parser.parse_args()
    
    
    mesh_params = mp.extract(args)
    device = "cuda:" + str(args.gpuID)
    
    test(mesh_params, device)
    