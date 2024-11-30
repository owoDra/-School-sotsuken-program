import utilities as utls
import convolution as conv
from position_embeddings import SinusoidalPositionEmbeddings
from normalization import PreNormalization
from attention import DotProductAttention, LinearAttention
from res_net import ResnetBlock

from functools import partial

import torch
from torch import nn


class Unet(nn.Module):
  def __init__(
    self,
    dim=32,
    init_dim=None,
    out_dim=None,  
    dim_mults=(1, 2, 4, 8),
    channels=3,
    with_time_emb=True,
    resnet_block_groups=8,
  ):
    super().__init__()
 
    self.channels = channels
    init_dim = utls.default(init_dim, dim // 3 * 2)
    self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
 
    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:])) # (input_dim, output_dim)というタプルのリストを作成する
 
    resnet_block = partial(ResnetBlock, groups=resnet_block_groups)
 
    # 時点情報埋め込み
    if with_time_emb:
      time_dim = dim
      # time_mlp: pos emb -> Linear -> GELU -> Linear
      self.time_mlp = nn.Sequential(
          SinusoidalPositionEmbeddings(dim),
          nn.Linear(dim, time_dim),
          nn.GELU(),
          nn.Linear(time_dim, time_dim)
      )
    else:
      time_dim = None
      self.time_mlp = None
 
    self.downs = nn.ModuleList([])
    self.ups = nn.ModuleList([])
    num_resolutions = len(in_out) # blockを処理する回数
 
    # ダウンサンプル
    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= (num_resolutions - 1)
 
      self.downs.append(
          nn.ModuleList(
              [
                  resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                  resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                  utls.Residual(PreNormalization(dim_out, LinearAttention(dim_out))),
                  conv.DownsampleConvolution(dim_out) if not is_last else nn.Identity(),
                
              ]
          )
      )
 
    # 中間ブロック
    mid_dim = dims[-1]
    self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
    self.mid_attn   = utls.Residual(PreNormalization(mid_dim, DotProductAttention(mid_dim)))
    self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
 
    # アップサンプル
    for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = ind >= (num_resolutions - 1)
 
      self.ups.append(
          nn.ModuleList(
              [
                resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
               resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
               utls.Residual(PreNormalization(dim_in, LinearAttention(dim_in))),
               conv.UpsampleConvolution(dim_in) if not is_last else nn.Identity(),
              ]
          )
      )
    out_dim = utls.default(out_dim, channels)
    self.final_conv = nn.Sequential(
        resnet_block(dim, dim),
        nn.Conv2d(dim, out_dim, 1)
    )

 
  def forward(self, x, time):
    x = self.init_conv(x)
    t = self.time_mlp(time) if utls.exists(self.time_mlp) else None
    h = []
 
    # ダウンサンプル
    for block1, block2, attn, downsample in self.downs:
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      h.append(x)
      x = downsample(x)
 
    # 中間
    x = self.mid_block1(x, t)
    x = self.mid_attn(x)
    x = self.mid_block2(x, t)
 
    # アップサンプル
    for block1, block2, attn, upsample in self.ups:
      x = torch.cat((x, h.pop()), dim=1) # downsampleで計算したhをくっつける
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      x = upsample(x)
 
    return self.final_conv(x)
  