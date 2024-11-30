import utilities as utls
import convolution as conv

import torch
from torch import nn

from einops import rearrange


class ResnetBlock(nn.Module):
  """
  　ResNet ブロック
  """
  def __init__(self, dim, dim_out, time_emb_dim = None, groups=8):
    super().__init__()

    # 時点情報を作成
    if utls.exists(time_emb_dim):
      self.mlp = (
          nn.Sequential(
          nn.SiLU(), 
          nn.Linear(time_emb_dim, dim_out)
          )
      )
    else:
      self.mlp = (None)
      
    # インプットとアウトプットのサイズが違う場合は畳み込み
    self.block1 = conv.ConvolutionBlock(dim, dim_out, groups=groups)
    self.block2 = conv.ConvolutionBlock(dim_out, dim_out, groups=groups)

    if dim != dim_out: 
      self.res_conv = nn.Conv2d(dim, dim_out, 1)
    else:
       self.res_conv = nn.Identity()

 
  def forward(self, x, time_emb=None):
    # 畳み込み 1
    h = self.block1(x) 
     
    # 時点情報の付加
    if utls.exists(self.mlp) and utls.exists(time_emb):
      time_emb = self.mlp(time_emb)
      h = rearrange(time_emb, "b c -> b c 1 1") + h 
 
    # 畳み込み 2
    h = self.block2(h)
     
    # 畳み込み + 残差結合
    return h + self.res_conv(x)
