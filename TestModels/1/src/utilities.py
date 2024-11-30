from inspect import isfunction

import torch
from torch import nn


def exists(x):
  """
    引数xがNoneでなければTrueを返し、NoneであればFalseを返す関数
  """
  return x is not None


def default(x, d):
  """
    xがNoneでなければTrueを返す.
    Noneの場合, dが関数であれば関数を呼び出した結果を返し, 関数でなければその値を返す.
  """
  if exists(x):
    return x
  return d() if isfunction(d) else d


def extract(a, t, x_shape):
  """
    配列から時点tに対応する要素配列を取得する
  """
  batch_size = t.shape[0]                                                     # バッチサイズ
  out = a.gather(-1, t.cpu())                                                 # aの最後の次元 ⇒ timestepに対応するalphaを取ってくる
  return out.reshape(batch_size, *((1,) * (len(x_shape) -  1))).to(t.device)  # バッチサイズ x 1 x 1 x 1にreshape


class Residual(nn.Module):
  """
    残差結合
  """
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
 
  def forward(self, x, *args, **kwargs):
    """
      f(x) + x
    """
    return self.fn(x, *args, **kwargs) + x