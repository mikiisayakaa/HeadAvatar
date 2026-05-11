# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import torch

PI = 3.1415926536

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        torch.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255*torch.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - torch.floor(255*torch.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = torch.floor(255*torch.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - torch.floor(255*torch.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = torch.floor(255*torch.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - torch.floor(255*torch.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (torch.Tensor): Itorchut horizontal flow of shape [H,W]
        v (torch.Tensor): Itorchut vertical flow of shape [H,W]

    Returns:
        torch.Tensor: Flow visualization image of shape [H,W,3]
    """
    flow_image = torch.zeros((u.shape[0], u.shape[1], 3), dtype=torch.uint8, device=u.device)
    colorwheel = make_colorwheel().to(u.device)  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = torch.sqrt(torch.square(u) + torch.square(v)).unsqueeze(-1)
    a = torch.arctan2(-v, -u)/PI
    fk = (a+1) / 2*(ncols-1)
    k0 = torch.floor(fk).to(torch.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = (fk - k0).unsqueeze(-1)
    col0 = colorwheel[k0] / 255.0
    col1 = colorwheel[k1] / 255.0
    col = (1-f)*col0 + f*col1
    idx = (rad <= 1).squeeze(-1)
    col[idx]  = 1 - rad[idx] * (1-col[idx])
    col[~idx] = col[~idx] * 0.75   # out of range
    return col


def flow_to_image(flow_uv):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (torch.Tensor): Flow UV image of shape [H,W,2]

    Returns:
        torch.ndarray: Flow visualization image of shape [H,W,3]
    """
    u = flow_uv[:,:,:,0]
    v = flow_uv[:,:,:,1]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    rad_max = torch.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v)
