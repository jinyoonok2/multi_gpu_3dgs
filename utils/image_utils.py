#
# Copyright (C) 2023, Inria GRAPHDECO research group
# Copyright (C) 2025, New York University
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE file.
#
# Original 3D Gaussian Splatting code from:
# https://github.com/graphdeco-inria/gaussian-splatting
#
# CLM-GS modifications by NYU Systems Group
# https://github.com/nyu-systems/CLM-GS
#

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
