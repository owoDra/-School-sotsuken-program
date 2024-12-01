import sys
sys.path.append("./src/")

from pathlib import Path

from src.References import *
from src.Train import train

# ======================================
Datasets = {
    DatasetRef("FewShotPokemon_833", "huggan/few-shot-pokemon", 500),
    # DatasetRef("PixelArtNouns_2000", "jiovine/pixel-art-nouns-2k", 500),
    # DatasetRef("FashionImage_3506", "GHonem/fashion_image_caption-3500", 500),
    # DatasetRef("Animals_5400", "mertcobanov/animals", 500),
    # DatasetRef("Landscapes", "mdroth/landscapes", 500)
}

Models = {
    ModelRef("GoogleDdpmCifar10", "google/ddpm-cifar10-32", "UNet2DModel", "DDPMScheduler"),
    ModelRef("GoogleDdpmCelebahq", "google/ddpm-celebahq-256", "UNet2DModel", "DDPMScheduler")
}

# ======================================

for dataset_ref in Datasets:
    for model_ref in Models:
        output_path = Path(str(f'./results/{dataset_ref.name}/{model_ref.name}/'))
        train(dataset_ref, model_ref, output_path, 1)
