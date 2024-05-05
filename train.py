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

from model import Unet
from utils import *
from data import Celeba

from pathlib import Path

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

results_folder = Path("./results/scratch/")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 200

model_save_dir = "./checkpoint_scratch/"


# parameters
dataroot = "./data/celeba/"
img_dir = dataroot + "image/"
annotation = dataroot + "anno/"
image_size = 28
channels = 3
batch_size = 128

# image transformations
transforms_a = T.Compose([
    T.ToTensor(),
    T.Resize((image_size, image_size)),
    T.RandomHorizontalFlip(),
    # T.Normalize((0.5), (0.5))# T.Normalize([0.5],[0.5])
])

transforms_b = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(), # turn into Numpy array of shape HWC, divide by 255
    T.Lambda(lambda t: (t * 2) - 1),
])

# Sampling Methods

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

reverse_transform = T.Compose([
    T.Lambda(lambda t: (t + 1) / 2),
    T.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    T.Lambda(lambda t: t * 255.),
    T.Lambda(lambda t: t.numpy().astype(np.uint8)),
    T.ToPILImage(),
])

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)     # t-1 ~ 0
        # print(type(img))   class tensor
        # imgs.append(img)
        img_cpu = img.cpu().numpy()
        # print(type(img))
        imgs.append(img_cpu)
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
# sample
# result: [(Batchsize, Channels(3), H, W) * Timesteps]


# loss

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    # 원본에 Noise를 넣어 Noisy한 Image를 만듦
    predicted_noise = denoise_model(x_noisy, t)
    # Noise 예측 

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# model 
if __name__ == "__main__":

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)
    model = nn.DataParallel(model)


    optimizer = Adam(model.parameters(), lr=1e-3)

    # training process

    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from data import Celeba
    from torch.utils.tensorboard import SummaryWriter

    dataset = Celeba(img_dir, None, transforms=transforms_b, mode="train")
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=16)

    epochs = 20

    print("train start!!")
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0: # batch size 128 기준 1epoch 당 약 1271 iteration
                print("Epoch: {0} | Step: {1:0>3} | Loss: {2:.4f}".format(epoch+1, step, loss.item()))
                # model save
                torch.save({
                    'batch_size': batch_size,
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, model_save_dir+"step{0:0>3}_model.tar".format(str(step)))

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(2, batch_size)  # 4 > batchsize 이면 2개 이상 
                all_images_list = list(map(lambda n: sample(model, image_size=image_size,batch_size=n, channels=channels), batches))
                # for _, img in enumerate(all_images_list):
                #     print("image type: {}".format(type(img)))
                #     print(img.size())
                # print(all_images_list)   

                all_images = np.concatenate(np.array(all_images_list), axis=0)
                # all_images = torch.cat(all_images_list, dim = 0)
                all_images = torch.tensor(all_images)
                
                # 특정 한 사진에 대해서 T에 대한 변화 저장
                for i in range(2):
                    images = torch.index_select(all_images, 1, torch.tensor([i])).squeeze()
                    # image = (timestamps, 3, img size, img size)
                    images = (images + 1) * 0.5
                    save_image(images, str(results_folder / f'epoch_{epoch+1}_sample-{milestone}-{i+1}th.png'), nrow = 8)