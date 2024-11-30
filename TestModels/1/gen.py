import sys
sys.path.append("./src/")

from pathlib import Path

from torchvision.utils import save_image

from safetensors.torch import load_model

from src.unet import Unet
from src.diffusion import *


# モデル読み込み
model_folder = Path("./trained_models/")
# model_name = str("diffusion_pytorch_model.fp16")
model_name = str("jiovine_pixel-art-nouns-2k-32x32-epochs10")

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 32
channels = 1

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    resnet_block_groups=4,
)
model.to(device)
load_model(model, str(model_folder / f'{model_name}.safetensors'))

# 画像の生成
results_folder = Path('./trained_results/')
results_folder.mkdir(exist_ok=True)

results_subfolder = Path(str(f'./trained_results/{model_name}/'))
results_subfolder.mkdir(exist_ok=True)

samples = sample(model, image_size=image_size, batch_size=1, channels=channels)

idx = 1
for smpl in samples:
    if idx % 20 == 0:
        save_image(torch.from_numpy(smpl), str(results_subfolder / f'timestep-{idx}.png'), nrow=5)
    idx += 1
