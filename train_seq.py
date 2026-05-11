#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import loss_utils as lu
from mesh_renderer_torch import render
import sys
from mscene import Scene, MeshModelBatch
from utils.general_utils import safe_state
from utils.image_utils import psnr, error_map, visualize_grid, get_semantic_image, imagespace_laplacian
import uuid
from tqdm import tqdm
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import MeshModelSequenceParams, TrainConfig
from utils.perceptual_loss import VGGPerceptualLoss
from utils.timer_utils import CudaTimer
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
def set_bits(mp, iteration, do_vertex, do_texture, do_semantic):
    do_vertex = False
    do_texture = False
    do_semantic = False
    if iteration >= mp.vertex_epoch[0] and iteration < mp.vertex_epoch[1]:
        do_vertex = True
    if iteration >= mp.texture_epoch[0] and iteration < mp.texture_epoch[1]:
        do_texture = True

    return do_vertex, do_texture, do_semantic

                
def training_mesh(mp, device):
    if mp.vhap:
        mp.texture_epoch = 0
        mp.epoch = 50
    torch.cuda.set_device(device)
    
    source_path = mp.source_path
    # id_path = os.path.join(source_path, "train_ids.txt")
    # mp.timesteps = []
    # with open(id_path, 'r') as f:
    #     for line in f:
    #         mp.timesteps.append(int(line.strip()))
    # mp.timesteps = [24]
    # mp.val_timesteps = [24]
    
    tb_writer = prepare_output_and_logger(mp)
    train_config = TrainConfig(mp)
    train_config.save_json(mp.model_path)
    
    scene = Scene(mp, device=device)
    scene.load_cameras(mp.timesteps)
    
    mesh_model = MeshModelBatch(args=mp, flame_info=scene.flame_info, device=device)

    first_iter = 0

    train_loader = DataLoader(scene.getTrainCameras(), mp.batch_size, shuffle=True,
                              collate_fn=scene.collate_fn)
    val_loader = DataLoader(scene.getValCameras(), 1, shuffle=False,
                            collate_fn=scene.collate_fn)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, mp.epoch), desc="Training progress")
    
    lr_boost = mp.batch_size
    do_texture = False
    do_vertices = True
    do_semantic = False
        
    timer = CudaTimer()
    
    VGGLoss = VGGPerceptualLoss().to(device)
    
    for epoch in range(first_iter, mp.epoch + 1):
        iter_start.record()
        
        do_vertices, do_texture, do_semantic = set_bits(mp, epoch, do_vertices, do_texture, do_semantic)

        for cam_batch in train_loader:
            with timer.record("Render: "):
                renderer = render(cam_batch, mesh_model, device) 
                
            with timer.record("Prepare images: "):
                image = torch.clamp(renderer.colors, 0.0, 1.0)
                mask = torch.clamp(renderer.mask, 0.0, 1.0)
                gt_image = cam_batch.original_image.to(device)
                gt_mask = cam_batch.original_mask.to(device)

            with timer.record("Losses: "):
                Ll1 = lu.l1_loss(image, gt_image) if do_texture else 0
                Ll1_mask = lu.l1_loss(mask, gt_mask)
                Lssim = (1.0 - lu.ssim(image, gt_image)) if do_texture else 0
                
                laplacian_reg = lu.laplacian_loss(renderer.vertices, mesh_model.laplacian) if mp.lambda_laplacian > 0 else 0
                smooth_reg = lu.normal_consistency_loss(renderer.vertices, mesh_model.mesh.faces, mesh_model.adjFaces) if mp.lambda_smooth > 0 else 0
                roughness_reg = lu.roughness_regularization(renderer.roughness, renderer.mask) if do_texture and mp.lambda_roughness > 0 else 0
                Lsemantic = lu.l1_loss(renderer.learnable_semantic, cam_batch.semantic)
                
                L_lpips = VGGLoss(image, gt_image, epoch) if do_texture else 0
                
                shader_reg = lu.shader_regularization(mesh_model.shader.adaptive, mesh_model.shader.neural_shader, renderer.texc[0], device, epoch) if do_texture and mp.neural else 0 
                roughness_tv = lu.tv_loss(renderer.roughness) if do_texture and mp.lambda_roughness_tv > 0 else 0
                
                loss = (mp.lambda_l1 * Ll1 + 
                        mp.lambda_dssim * Lssim +
                        mp.lambda_mask * Ll1_mask +
                        mp.lambda_laplacian * laplacian_reg +
                        mp.lambda_smooth * smooth_reg +
                        mp.lambda_roughness * roughness_reg +
                        mp.lambda_semantic * Lsemantic +
                        mp.lambda_shader_reg * shader_reg +
                        mp.lambda_roughness_tv * roughness_tv +
                        mp.lambda_lpips * L_lpips) * lr_boost

            with timer.record("Backward: "):
                loss.backward()

            with timer.record("Training update: "):
                with torch.no_grad():
                    # Optimizer step
                    if epoch <= mp.epoch:
                        mesh_model.training_update(do_vertices, do_texture, do_semantic)
        
        with timer.record("Logging: "):                       
            iter_end.record()
            with torch.no_grad():
                loss_dict = {
                    "l1": Ll1,
                    "ssim": (1.0 - lu.ssim(image, gt_image)),
                    "mask": Ll1_mask,
                    "semantic": Lsemantic,
                    "smooth": smooth_reg,
                    "laplacian": laplacian_reg,
                    "roughness": roughness_reg,
                    "lpips": L_lpips,
                    "roughness_tv": roughness_tv,
                    "shader_reg": shader_reg,
                    "total": loss / lr_boost
                }
                # Log and save
                training_report(tb_writer, val_loader, epoch, loss_dict, iter_start.elapsed_time(iter_end), mesh_model, device, mp)
            
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() / lr_boost + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}"})
                progress_bar.update(1)
                if epoch == mp.epoch:
                    progress_bar.close()
                    
            with torch.no_grad():
                if epoch == mp.epoch:
                    mesh_model.save(mp.model_path)
                    
    timer.summary()
                

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, val_loader, epoch, losses, elapsed, mesh_model, device, mp):
    if tb_writer:
        for loss_name, loss_val in losses.items():
            tb_writer.add_scalar('train_loss_patches/' + loss_name + "_loss", loss_val if isinstance(loss_val, int) else loss_val.item(), epoch)
        tb_writer.add_scalar('iter_time', elapsed, epoch)
        
    check = 50
        
    # visualization output
    if epoch % check == 0:
        save_image = epoch == mp.texture_epoch[0] or epoch == mp.epoch or epoch == 0
        # save_image = True
        val_criterion = {
            "l1" : 0,
            "psnr": 0,
            "ssim": 0,
            "lpips": 0,
        }
        for idx, camera in enumerate(val_loader):
            renderer = render(camera, mesh_model, device)
            semantic_gt = get_semantic_image(camera.semantic)
            head_mask = (semantic_gt.sum(dim=-1) > 0).unsqueeze(-1).float()
            normal_img = (renderer.normal + 1) / 2.0
            normal_img = normal_img * renderer.mask
            test_img = torch.clamp(renderer.colors, 0.0, 1.0)
            test_gt = camera.original_image.to(device)
            loss_img = test_img * head_mask
            loss_gt = test_gt * head_mask
            val_criterion["l1"] += lu.l1_loss(loss_img, loss_gt).double()
            val_criterion["psnr"] += psnr(loss_img, loss_gt).double()
            val_criterion["ssim"] += lu.ssim(loss_img, loss_gt).double()
            val_criterion["lpips"] += lpips(loss_img.permute(0, 3, 1, 2), loss_gt.permute(0, 3, 1, 2)).sum().double()
            
            if save_image:
                semantic_img = get_semantic_image(renderer.learnable_semantic)
                prefix = mp.model_path + "/" + str(epoch) + "_" + str(camera.timesteps[0]) + "t" + "_" + str(camera.camera_id[0]) + "c"
                error_img = error_map(test_img.squeeze().detach().cpu(), test_gt.squeeze().cpu())

                images_render = [
                    test_gt,
                    test_img,
                    error_img.to(test_img.device),
                    normal_img,
                    semantic_gt,
                    semantic_img,
                ]
            
                visualize_grid(images_render, nrow=3, save_path=prefix + "_render.png")
           
        for key in val_criterion:
            val_criterion[key] /= len(val_loader)
            tb_writer.add_scalar('val_loss_patches/' + key, val_criterion[key], epoch)
      
    torch.cuda.empty_cache()  
    
    return
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    mp = MeshModelSequenceParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpuID", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda:" + str(args.gpuID)
      
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=4)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(False)
    training_mesh(mp.extract(args), device)

    # All done
    print("\nTraining complete.")
