import sys
sys.path.append("./src/")
import os

from pathlib import Path

from src.References import *
from src.Train import train

# ======================================
Datasets = {
    DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 100),
    DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 500),
    DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 1000),
    DatasetRef("Animals_5400", "mertcobanov/animals", 100),
    DatasetRef("Animals_5400", "mertcobanov/animals", 500),
    DatasetRef("Animals_5400", "mertcobanov/animals", 1000)
}

Models = {
    ModelRef("GoogleDDPM", "google/ddpm-cifar10-32", "UNet2DModel", "DDPMScheduler"),
    ModelRef("UripperGIANNIS", "uripper/GIANNIS", "UNet2DModel", "DDPMScheduler"),
    ModelRef("JfjensenSD", "jfjensen/sd-class-butterflies-64", "UNet2DModel", "DDPMScheduler"),
    ModelRef("KsamlMnist", "ksaml/mnist-fashion_64", "UNet2DModel", "DDPMScheduler"),
    ModelRef("RafaelgDDPM", "rafaelg/ddpm-celebahq-finetuned-butterflies-2epochs", "UNet2DModel", "DDPMScheduler"),
    ModelRef("WiNE-iNEFF", "WiNE-iNEFF/Minecraft-Skin-Diffusion", "UNet2DModel", "DDPMScheduler"),
    ModelRef("TadisettirajuRaju", "tadisettiraju/raju_diffusion", "UNet2DModel", "DDPMScheduler"),
    ModelRef("MirasaraLM2", "Mirasara/lm2-class-wood-32", "UNet2DModel", "DDPMScheduler"),
    ModelRef("Likalto4", "Likalto4/Breast_unconditional_64", "UNet2DModel", "DDPMScheduler"),
    ModelRef("Myunus1", "myunus1/diffmodels_galaxies_scratchbook", "UNet2DModel", "DDPMScheduler"),
    ModelRef("Daspartho", "daspartho/bored-ape-diffusion", "UNet2DModel", "DDPMScheduler"),
    ModelRef("BenlehrburgerModern", "benlehrburger/modern-architecture-32", "UNet2DModel", "DDPMScheduler"),
}

# ======================================

for dataset_ref in Datasets:
    for model_ref in Models:
        output_path = Path(str(f'./UnconditionalImageGen/results/{dataset_ref.name}_{dataset_ref.row}/{model_ref.name}/'))

        if os.path.exists(output_path):
            print(str(f'[{output_path}] already exist; skip'))
            continue

        train(dataset_ref, model_ref, output_path)
