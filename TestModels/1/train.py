import sys
sys.path.append("./src/")

from datasets import load_dataset, config

import numpy as np
 
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
 
from torch.optim import Adam
 
from pathlib import Path

from safetensors.torch import save_model

from src.unet import Unet
from src.diffusion import *

# dataset_name = str("zalando-datasets/fashion_mnist")
# dataset_name = str("svjack/pokemon-blip-captions-en-zh")
# dataset_name = str("huggan/few-shot-pokemon")
dataset_name = str("jiovine/pixel-art-nouns-2k")

config.DOWNLOADED_DATASETS_PATH = Path("./datasets/")
dataset = load_dataset(dataset_name, trust_remote_code=True)

image_size = 32
channels = 1
batch_size = 1

transform = Compose([
    transforms.Resize(size=(32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def transforms_data(examples):
  # 画像データを数値データに変換
  examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
  del examples["image"]
  return examples
 
transformed_dataset = dataset.with_transform(transforms_data).remove_columns("text")
# transformed_dataset = dataset.with_transform(transforms_data).remove_columns("label")
# transformed_dataset = dataset.with_transform(transforms_data).remove_columns("en_text").remove_columns("zh_text")
 
# データローダーの作成
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

# 生成した画像を保存するためのフォルダを設定
results_folder = Path("./results/")
results_folder.mkdir(exist_ok=True)

# モデルの作成
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    resnet_block_groups=4,
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

# 学習

epochs = 10

for epoch in range(epochs):
  for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
 
    batch_size = batch["pixel_values"].shape[0]
    batch = batch["pixel_values"].to(device)

    # タイムステップ情報をバッチごとにランダムに与える
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()

    # 画像を生成し損失を計算
    loss = p_losses(model, batch, t)
 
    # 損失を表示
    if step % 100 == 0: 
      print(str(f'epoch: {epoch} | Loss'), loss.item())
 
    # パラメータの更新
    loss.backward()
    optimizer.step()
 
  # 画像の生成
  samples = sample(model, image_size=image_size, batch_size=1, channels=channels)
  idx = 1
  for smpl in samples:
    if idx % 200 == 0:
      save_image(torch.from_numpy(smpl), str(results_folder / f'sample-{epoch}.png'), nrow=5)
    idx += 1

# モデル保存
model_folder = Path("./trained_models/")
model_folder.mkdir(exist_ok=True)

dataset_name = dataset_name.replace('/', '_')

save_model(model, str(model_folder / f'{dataset_name}-{image_size}x{image_size}-epochs{epochs}.safetensors'))