import torch


def getLightDir(W,H,device="cpu"):
    x = torch.arange(0,W,device=device)
    y = torch.arange(0,H,device=device)
    if True:
        y = H - 1 - y
    grid_y,grid_x = torch.meshgrid(y,x)
    u = (grid_x + 0.5) / W
    v = (grid_y + 0.5) / H
    phi = u * torch.pi * 2
    theta = (v - 0.5) * torch.pi
    y = torch.sin(theta)
    r = torch.cos(theta)
    x = r * torch.cos(phi)
    z = r * torch.sin(phi)
    return torch.stack([x,y,z],dim=0) # 3xHxW
