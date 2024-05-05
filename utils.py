import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from data import Celeba
from torchvision import transforms as T

# use seed for reproducability
torch.manual_seed(0)

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
# def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0]) + with_orig
#     fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         row = [image] + row if with_orig else row
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     if with_orig:
#         axs[0, 0].set(title='Original image')
#         axs[0, 0].title.set_size(8)
#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])

#     plt.tight_layout()
    


# beta scheduling 
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# forward util
timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)
# (timestamp,)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)  


alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # 맨 뒷부분 하나 자르고 앞부분에 1 추가

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)    # Backward Process 시 사용

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)


sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # Backward Process 시 사용
def extract(a, t, x_shape): # batch index에 대한 적절한 T를 추출하는 방법
    # a: 원하는 변수 ex) beta, alpha 등
    # t: (batch_size, ) 0 ~ timestamp 사이 정수 중 랜덤하게 Batch Size 만큼 추출한 것
    # t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    # x_shape: (batch_size, channels, H, W)
    # ex) extract(sqrt_alphas_cumprod, t, x_start.shape)
    
    # F.pad() 의 의미 
    
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) 
    # t = [timestamp_1, timestamp_2, ..., timestamp_t]
    # a = [a_1, a_2, ... , a_timesteps]
    # a 중에서 각 t의 원소, 즉 time stamp에 해당하는 값들을 뽑는다.
    
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)   # N C H W 형식 맞추려고 뒤에 차원 추가하여 GPU load

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

    
