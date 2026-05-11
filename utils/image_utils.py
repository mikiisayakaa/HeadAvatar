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

import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
from matplotlib import cm
import torchvision.utils as vutils
import cv2
from sklearn.cluster import KMeans

CMAP = torch.tensor(
        [
            (0, 0, 0),
            (255, 255, 0),
            (139, 76, 57),
            (255, 54, 38),
            (0, 205, 0),
            (0, 138, 0),
            (154, 50, 205),
            (72, 118, 255),
            (255, 165, 0),
        ],
        dtype=torch.float32
    ) / 255.0

CMAP2 = torch.tensor(
        [
            (0, 0, 0),
            (255, 255, 0),
            (139, 76, 57),
            (255, 54, 38),
            (0, 205, 0),
            (0, 138, 0),
            (154, 50, 205),
            (72, 118, 255),
            (255, 165, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ],
        dtype=torch.float32
    ) / 255.0

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def imagespace_laplacian(img):
    # input image of shape (B, H, W, C)
    temp_img = img.permute(0, 3, 1, 2).contiguous()
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(temp_img.shape[1], -1, -1, -1)
    laplacian = F.conv2d(temp_img, kernel, padding=1, groups=temp_img.shape[1])
    laplacian = laplacian.permute(0, 2, 3, 1).contiguous()
    return laplacian
    
def image2numpy(image):

    if len(image.shape) == 4:
        img = image.squeeze(0).detach().cpu()
    else:
        img = image.detach().cpu()
    
    img = img.clamp(0, 1)
    
    img = (img * 255).to(torch.uint8).numpy()
    
    return img

def export_img(dir, img):
    # convert from RGB to BGR
    img = img[:, :, ::-1]
    import cv2
    cv2.imwrite(dir, img)
    
def load_img(dir):
    img = cv2.imread(dir)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(torch.float32) / 255.0
    return img
    
def make_gif(dir, imgs):
    ones = np.ones((imgs[0].shape[0], imgs[0].shape[1], 1), dtype=np.uint8) * 255
    alpha_images = [np.concatenate((img, ones), axis=-1) for img in imgs]
    images = [Image.fromarray(img) for img in alpha_images]
    
    images[0].save(
        dir,
        save_all=True,
        append_images=images[1:],
        duration=200,             
        loop=1                    
    )

def error_map(img1, img2):
    error = (img1 - img2).mean(dim=2) / 2 + 0.5
    cmap = cm.get_cmap("seismic")
    error_map = cmap(error.cpu())
    return torch.from_numpy(error_map[..., :3]).unsqueeze(0)


def visualize_grid(images, nrow, save_path):
    for idx in range(len(images)):
        if images[idx].shape[-1] in [1, 3]: 
            images[idx] = images[idx].squeeze(0).permute(2, 0, 1)
    
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True, value_range=(0, 1))
    
    grid_np = image2numpy(grid.permute(1, 2, 0))
    
    export_img(save_path, grid_np)
    
def get_semantic_image(semantic):
    semantic_color = semantic.view(-1, CMAP2.shape[0]) @ (CMAP2.to(semantic.device))
    return semantic_color.view(semantic.shape[0], semantic.shape[1], semantic.shape[2], 3)
  

def generate_checkerboard(size=8, num_tiles=8):
    row_pattern = torch.tensor([0, 1] * (num_tiles // 2), dtype=torch.float32)
    checkerboard_row = row_pattern.repeat_interleave(size)
    board = []

    for i in range(num_tiles):
        row = checkerboard_row if i % 2 == 0 else 1 - checkerboard_row
        board.extend([row] * size)

    checkerboard = torch.stack(board)
    return checkerboard

def sample_region_main_color(image, mask, resize=0.5):
    # input numpy array of shape (H, W, 3)
    img = cv2.resize(image, (int(image.shape[1] * resize), int(image.shape[0] * resize)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (int(image.shape[1] * resize), int(image.shape[0] * resize)), interpolation=cv2.INTER_AREA)
    colors = img[mask > 0.5]
    
    main_color = sample_main_color(colors)
    return main_color

def sample_main_color(colors):
    # input numpy array of shape (N, 3)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(colors)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_label = labels[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_label]
    
    return dominant_color[0]
