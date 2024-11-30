import sys
sys.path.append("./src/")

from datasets import load_dataset, config
from torchvision import transforms
from pathlib import Path
import torch

from src.References import *

def get_dataset(url):
    config.DOWNLOADED_DATASETS_PATH = Path("./datasets/")
    return load_dataset(url, trust_remote_code=True)

def resolve_dataset(dataset, image_size, channels, row):
    transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def transforms_data(examples):
        # 画像データを数値データに変換
        images = [transform(image.convert("L")) for image in examples["image"]] if (channels == 1) else [transform(image.convert("RGB")) for image in examples["image"]]
        if row > 0: 
            images = images[0:row]
        return {"input": images}
    
    return dataset.with_transform(transforms_data)
   
def make_dataloader(url, row, image_size, channels, train_batch_size, dataloader_num_workers):
    dataset = resolve_dataset(get_dataset(url), image_size, channels, row)
    return torch.utils.data.DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=True, num_workers=dataloader_num_workers)
