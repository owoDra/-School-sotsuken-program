import math

import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
  """
    正弦波を用いた時点情報の埋め込み用クラス
  """
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
 
  def forward(self, time):
    device      = time.device 
    half_dim    = self.dim // 2 # 次元の半分
    embeddings  = math.log(10000) / (half_dim - 1) 
    embeddings  = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings  = time[:, None] * embeddings[None, :]
    embeddings  = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings
