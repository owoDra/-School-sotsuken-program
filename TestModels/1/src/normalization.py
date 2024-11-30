import torch
from torch import nn

class PreNormalization(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
 
  def forward(self, x):
    x = self.norm(x)
    return self.fn(x)
  