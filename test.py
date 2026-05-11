import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import loss_utils as lu
from mesh_renderer_torch import render
from mscene import Scene, MeshModelTest
from utils.general_utils import safe_state
from utils.image_utils import psnr, error_map, visualize_grid, get_semantic_image, imagespace_laplacian
from utils.perceptual_loss import VGGPerceptualLoss
from tqdm import tqdm
from lpipsPyTorch import lpips
from argparse import ArgumentParser
from arguments import MeshModelTestParams
from utils.timer_utils import CudaTimer
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
def test(mp, device):
    torch.cuda.set_device(device)
    
    scene = Scene(mp, device)
    scene.load_cameras_test(mp.timesteps)
    mesh_model = MeshModelTest(mp, scene.flame_info, device)
    
    first_iter = 0 
    
    val_criterion = {
        "l1" : 0,
        "psnr": 0,
        "ssim": 0,
        "lpips": 0,
    }
    
    if args.train:
        save_path = os.path.join(mp.load_path, "test_results_train_2")
    else:
        save_path = os.path.join(mp.load_path, "test_results_near")
    os.makedirs(save_path, exist_ok=True)
    
    VGGLoss = VGGPerceptualLoss().to(device)
    timer = CudaTimer()

    for timestep in mp.timesteps:
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, mp.epoch), desc="Training progress")
        # print(f"Testing timestep {timestep}")
        train_loader = DataLoader(scene.getTestTrainCameras(timestep), 4, shuffle=True, collate_fn=scene.collate_fn)
        val_loader = DataLoader(scene.getTestValCameras(timestep), 1, shuffle=False, collate_fn=scene.collate_fn)

        if args.train:
            for epoch in range(first_iter, mp.epoch):
                cam_batch = next(iter(train_loader))
                renderer = render(cam_batch, mesh_model, device)
                
                image = torch.clamp(renderer.colors, 0.0, 1.0)
                mask = torch.clamp(renderer.mask, 0.0, 1.0)
                gt_image = cam_batch.original_image.to(device)
                gt_mask = cam_batch.original_mask.to(device)
            
                Ll1 = lu.l1_loss(image, gt_image)
                Ll1_mask = lu.l1_loss(mask, gt_mask)
                Lssim = (1.0 - lu.ssim(image, gt_image))
                Lsemantic = lu.l1_loss(renderer.learnable_semantic, cam_batch.semantic)
                L_lpips = VGGLoss(image, gt_image, epoch) if mp.lambda_lpips > 0.0 else 0

                loss = (mp.lambda_l1 * Ll1 + 
                        mp.lambda_dssim * Lssim +
                        mp.lambda_mask * Ll1_mask +
                        mp.lambda_semantic * Lsemantic + 
                        mp.lambda_lpips * L_lpips)
                
                loss.backward()
                
                if epoch < mp.epoch:
                    with torch.no_grad():
                        mesh_model.training_update()
                        
                with torch.no_grad():
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}"})
                    progress_bar.update(1)
                    if epoch == mp.epoch:
                        progress_bar.close()
                        
        # test result

        with torch.no_grad():
            for cam_batch in val_loader:
                with timer.record("render:"):
                    renderer = render(cam_batch, mesh_model, device)
                
                test_img = torch.clamp(renderer.colors, 0.0, 1.0)
                test_gt = cam_batch.original_image.to(device)
                semantic_img = get_semantic_image(renderer.learnable_semantic)
                normal_img = (renderer.normal + 1) / 2.0
                normal_img = normal_img * renderer.mask
                
                semantic_gt = get_semantic_image(cam_batch.semantic)
                head_mask = (semantic_gt.sum(dim=-1) > 0).unsqueeze(-1).float()
                loss_img = test_img * head_mask
                loss_gt = test_gt * head_mask
                val_criterion["l1"] += lu.l1_loss(loss_img, loss_gt).item()
                val_criterion["psnr"] += psnr(loss_img, loss_gt).item()
                val_criterion["ssim"] += lu.ssim(loss_img, loss_gt).item()
                val_criterion["lpips"] += lpips(loss_img.permute(0, 3, 1, 2), loss_gt.permute(0, 3, 1, 2)).sum().item()

                # images = [test_gt, test_img, normal_img, semantic_gt, semantic_img]
                images = [test_img]
                # if cam_batch.camera_id[0] == 8:
                visualize_grid(images, 1, os.path.join(save_path, f"{str(timestep).zfill(5)}_{str(cam_batch.camera_id[0]).zfill(2)}.png"))
    timer.summary()
    
    # with open(os.path.join(save_path, "test_results2.txt"), "w") as f:
    #     num_val = len(mp.test_val_ids)
    #     for key in val_criterion:
    #         val_criterion[key] /= (num_val * len(mp.timesteps))
    #         f.write(f"{key}: {val_criterion[key]}\n")
    #         print(f"{key}: {val_criterion[key]}")
        

            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpuID", type=int, default=0)
    mp = MeshModelTestParams(parser)

    args = parser.parse_args()
    
    
    mesh_params = mp.extract(args)
    device = "cuda:" + str(args.gpuID)
    
    # test_ids_path = os.path.join(args.source_path, "test_ids.txt")
    # test_ids = []
    # with open(test_ids_path, "r") as f:
    #     for line in f:
    #         test_ids.append(int(line.strip()))
    # mesh_params.timesteps = test_ids
    # print("Test timesteps:", mesh_params.timesteps)
    # mesh_params.timesteps = [0]
    
    test(mesh_params, device)
    