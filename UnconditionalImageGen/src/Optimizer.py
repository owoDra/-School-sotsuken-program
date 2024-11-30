import sys
sys.path.append("./src/")

import torch

# 
# @Return Optimizer
# 
def get_optimizer(params, lr = 1e-4, betas = (0.95, 0.999), weight_decay = 1e-6, eps = 1e-08):
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )