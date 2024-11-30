import sys
sys.path.append("./src/")

from pathlib import Path
import torch
import torch.nn.functional as F

from src.Model import *
from src.References import *
from src.Dataset import make_dataloader
from src.Unet import resolve_unet
from src.NoiseScheduler import resolve_noise_scheduler
from src.Optimizer import get_optimizer
from src.LRScheduler import get_lr_scheduler
from src.Pipeline import resolve_pipeline

def train(
        dataset_ref:DatasetRef
        , model_ref:ModelRef
        , output_path:Path
        , num_epochs = 10
        , train_batch_size = 16
        , num_works = 0):
    output_path.mkdir(parents=True, exist_ok=True)

    # モデルの作成
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_classname, scheduler_classname = get_ref_class_names_from_model(model_ref.url)
    unet_classname = model_ref.unet_classname
    scheduler_classname = model_ref.scheduler_classname
    model, image_size, channels = resolve_unet(unet_classname, model_ref.url, device)

    # データローダー作成
    dataloader = make_dataloader(dataset_ref.url, dataset_ref.row, image_size, channels, train_batch_size, num_works)

    # ノイズスケジューラーの作成
    noise_scheduler = resolve_noise_scheduler(scheduler_classname, model_ref.url)

    # オプティマイザーの作成
    optimizer = get_optimizer(model.parameters())

    # lrスケジューラーの作成
    lr_scheduler = get_lr_scheduler(optimizer, len(dataloader) * num_epochs)

    # --------------------------------------------
    # モデルの学習
    # --------------------------------------------

    weight_dtype = torch.float16

    for epoch in range(0, num_epochs):
        model.train()

        for step, batch in enumerate(dataloader):
            batch_size = batch["input"].shape[0]
            clean_images = batch["input"].to(device)

            # 画像に適用するノイズを生成
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=device)

            # ランダムなタイムステップを生成
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

            # 学習用画像にノイズを乗せる
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # ノイズから画像を復元
            model_output = model(noisy_images, timesteps).sample
            
            # 学習元画像と生成画像の差を出力
            loss = F.mse_loss(model_output.float(), noise.float())
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            print(str(f'epoch: {epoch} | Loss'), loss.item())

    # pipeline に変換して保存
    pipeline = resolve_pipeline(scheduler_classname, model, noise_scheduler)
    pipeline.save_pretrained(output_path)
    print(str(f'{output_path} saved'))