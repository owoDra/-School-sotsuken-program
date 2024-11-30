import utilities as utls

import numpy as np
from tqdm.auto import tqdm
 
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


def linear_beta_schedule(timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 200
betas = linear_beta_schedule(timesteps=timesteps)
	
alphas = 1. - betas

alphas_cumprod = torch.cumprod(alphas, axis=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def q_sample(x_start, t, noise=None):
  """
  キレイな画像からノイズを加えた画像をサンプリングする.
  """
  if noise is None:                     # 呼び出し元からノイズが渡されていなければここでで生成する.
    noise = torch.randn_like(x_start)   # 正規乱数
 
  # t時点の平均計算用
  sqrt_alphas_cumprod_t = utls.extract(sqrt_alphas_cumprod, t, x_start.shape) 
  # t時点の標準偏差計算用
  sqrt_one_minus_alphas_cumprod_t = utls.extract(
      sqrt_one_minus_alphas_cumprod, t, x_start.shape
  )
 
  # (5)式で計算
  return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None):
  if noise is None:
    noise = torch.randn_like(x_start)
 
  x_noisy = q_sample(x_start=x_start, t=t, noise=noise) # x_tを計算
  predicted_noise = denoise_model(x_noisy, t) # モデルでノイズを予測
 
  loss = F.l1_loss(noise, predicted_noise) # 損失を計算
   
  return loss  


alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 標準偏差


def p_sample(model, x, t, t_index):
  # beta_t
  betas_t = utls.extract(betas, t, x.shape)
  # 1 - √\bar{α}_t
  sqrt_one_minus_alphas_cumprod_t = utls.extract(
      sqrt_one_minus_alphas_cumprod, t, x.shape
  )
  # 1 / √α_t
  sqrt_recip_alphas_t = utls.extract(sqrt_recip_alphas, t, x.shape)
 
  # μ_Θをモデルで求める: model(x, t)
  model_mean = sqrt_recip_alphas_t * (
      x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
  )
 
  if t_index == 0:
    return model_mean
  else:
    posterior_variance_t = utls.extract(posterior_variance, t, x.shape) # σ^2_tを計算
    noise = torch.randn_like(x) # 正規乱数zをサンプリング
 
  return model_mean + torch.sqrt(posterior_variance_t) * noise # x_{t-1}


@torch.no_grad()
def p_sample_loop(model, shape):
  device = next(model.parameters()).device
 
  b = shape[0]
  img = torch.randn(shape, device=device)
  imgs = []
 
  for i in tqdm(reversed(range(0, timesteps)), total=timesteps):
    img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    imgs.append(img.cpu().numpy())
  return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
  return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
