import torch
from torch import nn


class UpsampleConvolution(nn.Module):
  """
    アップサンプル用の畳み込み層
  """
  def __init__(self, dim):
    super().__init__()
    self.trans_conv = nn.ConvTranspose2d(
      in_channels   = dim, 
      out_channels  = dim, 
      kernel_size   = 4, 
      stride        = 2, 
      padding       = 1 
    )
 
  def forward(self, x):
    return self.trans_conv(x)
  

class DownsampleConvolution(nn.Module):
  """
  　ダウンサンプル用の畳み込み層
  """
  def __init__(self, dim):
    super().__init__()
    self.conv = nn.Conv2d(
      in_channels=dim, 
      out_channels=dim, 
      kernel_size=4, 
      stride=2, 
      padding=1
    )
 
  def forward(self, x):
    return self.conv(x)
  

class ConvolutionBlock(nn.Module):
  """
  　残差結合ブロックで使う畳み込みブロック
  """
  def __init__(self, dim, dim_out, groups=8):
    super().__init__()
    self.conv = nn.Conv2d(dim, dim_out, 3, padding=1) # 畳み込み層
    self.norm = nn.GroupNorm(groups, dim_out)         # 正規化層
    self.act  = nn.SiLU()                             # 活性関数
 
  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.act(x)
    return x
  