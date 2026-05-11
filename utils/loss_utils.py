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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils import mesh_utils as mu

class UncertaintyLoss(torch.nn.Module):
    def __init__(self, loss_num, coeff, lr=0.01):
        super(UncertaintyLoss, self).__init__()
        self.loss_num = loss_num
        self.coeff = coeff
        self.log_vars = torch.nn.Parameter(torch.zeros((loss_num)), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.log_vars], lr=lr)
        
    def forward(self, losses):
        assert len(losses) == self.loss_num
        total_loss = 0
        for i in range(self.loss_num):
            precision = 2 * torch.exp(-self.log_vars[i])
            total_loss += precision * losses[i] + self.log_vars[i]
        return total_loss * self.coeff
    
    def training_update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        

def l1_loss(pred, gt):
    return torch.abs((pred - gt)).mean()

def l1_loss_semantic(pred, gt, mask):
    mask = mask.view(-1, 1, 1, 1)
    # all zero
    if mask.sum() < 0.5:
        return torch.tensor(0, dtype=torch.float32, device=pred.device)
    return torch.abs((pred * mask - gt * mask)).sum() / (pred.shape[1] * pred.shape[2] * pred.shape[3] * mask.sum())

def entropy_loss(img):
    p = torch.softmax(img, dim=-1)
    entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    return entropy.mean()

def sum_loss(img):
    sum = torch.sum(img, dim=-1)
    return ((sum - 1.0) ** 2).mean()

def range_loss(img):
    # 小于0的惩罚
    loss_neg = torch.clamp(-img, min=0.0)
    # 大于1的惩罚
    loss_pos = torch.clamp(img - 1.0, min=0.0)
    return (loss_neg + loss_pos).mean()

def tv_loss(img):
    assert(len(img.shape) == 4)
    diff_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_y = img[:, 1:, :, :] - img[:, :-1, :, :]
    return (diff_x.abs().mean() + diff_y.abs().mean())

def gaussian_blur(img, kernel_size=5, sigma=1.0):
    channels = img.shape[1]
    # Create Gaussian kernel
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=img.device)
    x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    padding = kernel_size // 2
    img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
    blurred = F.conv2d(img_padded, kernel, groups=channels)
    return blurred

def normal_loss(pred, gt):
    # high frequency component loss
    pred_blur = gaussian_blur(pred, kernel_size=3)
    gt_blur = gaussian_blur(gt, kernel_size=3)
    high_freq_pred = pred - pred_blur
    high_freq_gt = gt - gt_blur
    # for i in range(pred.shape[0]):
    #     high_freq_1d = torch.linalg.norm(high_freq_gt[i].permute(1, 2, 0), dim=-1)
    #     high_freq_1d = torch.clamp(high_freq_1d * 10, 0, 1)
    #     high_freq_1d = high_freq_1d.unsqueeze(-1).repeat(1, 1, 3)
    #     img = iu.image2numpy(high_freq_1d)
    #     iu.export_img("normal_highfreq_" + str(i) + ".png", img)
    # import pdb
    # pdb.set_trace()
    return l1_loss(high_freq_pred, high_freq_gt)

def l2_loss(pred, gt):
    return ((pred - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def laplacian_loss(vertices, laplacian):
    batch_laplacian = torch.stack([laplacian] * vertices.shape[0], dim=0)
    loss = torch.bmm(batch_laplacian, vertices)
    loss = loss.norm(dim=-1)**2
    
    return loss.mean()

def angle_loss(vertices, faces, gt_angles):
    angles = mu.compute_triangle_angles(vertices, faces)
    return l1_loss(angles, gt_angles[None, ...])


def normal_consistency_loss(vertices, faces, adj_faces):
    """ Compute the normal consistency term as the cosine similarity between neighboring face normals.

    Args:
        mesh (Mesh): Mesh with face normals.
    """
    face_normals = mu.compute_face_normals(vertices, faces)
    loss = 1 - torch.cosine_similarity(face_normals[:, adj_faces[:, 0], :], face_normals[:, adj_faces[:, 1], :], dim=-1)
    return (loss**2).mean()

def white_light_regularization(shading):
    white = torch.sum(shading, dim=-1, keepdims=True) / 3.0
    loss = torch.mean(torch.abs(shading - white))
    return loss

def shader_regularization(_adaptive, shader, uvs, device, iteration=0):

    pe_input = shader.apply_pe(uvs)
    kd = shader.material_mlp(pe_input)

    # add jitter for loss function
    jitter_uvs = uvs + torch.normal(mean=0, std=0.001, size=uvs.shape, device=device)
    jitter_pe_input = shader.apply_pe(jitter_uvs)

    kd_jitter = shader.material_mlp(jitter_pe_input)
    loss_fn = torch.nn.MSELoss(reduction='none')
    kd_grad = loss_fn(kd_jitter, kd)
    loss = torch.mean(_adaptive.lossfun(kd_grad.view(-1, 4)))
    return loss * min(1.0, iteration / 200)


def roughness_regularization(roughness, mask):
    mean_rough = 0.5 # 0.4 default
    std_rough = 0.1
    z_score = (roughness * mask - mean_rough) / std_rough

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss

def specular_regularization(rho, mask):
    # pre-computed
    mean_rough = 0.3753
    std_rough = 0.1655
    z_score = (rho * mask - mean_rough) / std_rough

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss

def landmark_loss(pred, gt, gt_val, width, height):
    # mask = landmark_mask.reshape(-1, 1, 1)
    # if mask.sum() < 0.5:
    #     return torch.tensor(0, dtype=torch.float32, device=pred.device)
    normalized_vec = gt.to(pred.device) - pred
    normalized_vec[..., 0] /= width
    normalized_vec[..., 1] /= height
    
    # loss = torch.sum(torch.norm(normalized_vec, dim=-1) * gt_val.to(pred.device) * mask)
    loss = torch.sum(torch.norm(normalized_vec, dim=-1) * gt_val.to(pred.device))
    
    return loss



