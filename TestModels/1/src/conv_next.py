import utilities as utls
import convolution as conv

import torch
from torch import nn

from einops import rearrange


class ConvNextBlock(nn.Module):
    """
  　  ConvNeXTブロック
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()

        # 時点情報を作成
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if utls.exists(time_emb_dim)
            else None
        )

        # 畳み込み層を作成
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None):
        # 
        h = self.ds_conv(x)

        # 時点情報の付加
        if utls.exists(self.mlp) and utls.exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        # 
        h = self.net(h)

        # 
        return h + self.res_conv(x)
    