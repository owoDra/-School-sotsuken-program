import sys
sys.path.append("./src/")

from src.unet import Unet
import torch


image_size = 128
channels = 3
batch_size = 8
timesteps = 200

model = Unet(
    dim=image_size,
    dim_mults=(1, 2, 4, 8),
    channels=channels,
    with_time_emb=True,
    resnet_block_groups=2,
)

data = torch.randn((batch_size, channels, image_size, image_size))
t = torch.randint(0, timesteps, (batch_size,)).long()

output = model(data, t)

print(output.size())
