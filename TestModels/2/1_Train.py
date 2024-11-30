import sys
sys.path.append("./src/")

from datasets import load_dataset, config

import torch
from torchvision import transforms
 
import torch.nn.functional as F

from pathlib import Path

from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler

import Arg

# --------------------------------------------
# データセット読み込み
# --------------------------------------------

config.DOWNLOADED_DATASETS_PATH = Path("./datasets/")
dataset = load_dataset(Arg._NAME_OR_PATH_DATASET_, trust_remote_code=True)

# --------------------------------------------
# データセット読み込み
# --------------------------------------------

image_size = 32
channels = 1
batch_size = 1

unet_config = UNet2DModel.load_config(Arg._NAME_OR_PATH_MODEL_)
image_size = unet_config["sample_size"]
channels = unet_config["out_channels"]

# --------------------------------------------
# データセットを加工
# --------------------------------------------

transform = transforms.Compose([
    transforms.Resize(size=(image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def transforms_data(examples):
  # 画像データを数値データに変換
  images = [transform(image.convert("L")) for image in examples["image"]] if (channels == 1) else [transform(image.convert("RGB")) for image in examples["image"]]
  return {"input": images}

# dataset.set_transform(transforms_data)
transformed_dataset = dataset.with_transform(transforms_data)
 
dataloader = torch.utils.data.DataLoader(
        transformed_dataset["train"], batch_size=Arg.train_batch_size, shuffle=True, num_workers=Arg.dataloader_num_workers
    )

# 生成した画像を保存するためのフォルダを設定
results_folder = Path("./results/")
results_folder.mkdir(exist_ok=True)

# --------------------------------------------
# モデルの作成
# --------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet2DModel.from_config(unet_config)
model.to(device)

# --------------------------------------------
# ノイズスケジューラーの作成
# --------------------------------------------

noise_scheduler_config = DDPMScheduler.load_config(Arg._NAME_OR_PATH_MODEL_)
noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

# --------------------------------------------
# オプティマイザーの作成
# --------------------------------------------

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Arg.lr_rate,
        betas=(Arg.adam_beta1, Arg.adam_beta2),
        weight_decay=Arg.adam_weight_decay,
        eps=Arg.adam_epsilon,
    )

# --------------------------------------------
# lrスケジューラーの作成
# --------------------------------------------

lr_scheduler = get_scheduler(
        Arg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=Arg.lr_warmup_steps * Arg.gradient_accumulation_steps,
        num_training_steps=(len(dataloader) * Arg.num_epochs),
    )

# --------------------------------------------
# モデルの学習
# --------------------------------------------

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

global_step = 0
first_epoch = 0

weight_dtype = torch.float16

for epoch in range(first_epoch, Arg.num_epochs):
    model.train()

    for step, batch in enumerate(dataloader):
        batch_size = batch["input"].shape[0]
        clean_images = batch["input"].to(device)

        noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=device)
        
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        model_output = model(noisy_images, timesteps).sample

        if Arg.prediction_type == "epsilon":
            loss = F.mse_loss(model_output.float(), noise.float()) 
        elif Arg.prediction_type == "sample":
            alpha_t = _extract_into_tensor(
                noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
            )
            snr_weights = alpha_t / (1 - alpha_t)
            
            loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
            loss = loss.mean()
        else:
            raise ValueError(f"Unsupported prediction type: {Arg.prediction_type}")
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        print(str(f'epoch: {epoch} | Loss'), loss.item())

    if epoch % Arg.save_model_epochs == 0 or epoch == Arg.num_epochs - 1:

        pipeline = DDPMPipeline(
            unet=model,
            scheduler=noise_scheduler,
        )

        pipeline.save_pretrained(Arg.output_dir)
