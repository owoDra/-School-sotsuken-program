import sys
sys.path.append("./src/")

from pathlib import Path

from src.References import *
from src.Train import train

# ======================================
Datasets = {
    # DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 10),
    DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 100),
    DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 500),
    DatasetRef("PixelArtNouns", "jiovine/pixel-art-nouns-2k", 1000),
    # DatasetRef("FewShotPokemon_833", "huggan/few-shot-pokemon", 500),
    # DatasetRef("FashionImage_3506", "GHonem/fashion_image_caption-3500", 500),
    # DatasetRef("Animals_5400", "mertcobanov/animals", 500),
    # DatasetRef("Landscapes", "mdroth/landscapes", 500)
}

Models = {
    ModelRef("GoogleDdpm", "google/ddpm-cifar10-32", "UNet2DModel", "DDPMScheduler"),
    ModelRef("JfjensenSD", "jfjensen/sd-class-butterflies-32", "UNet2DModel", "DDPMScheduler"),
    ModelRef("CyantifiCQNoisy", "CyantifiCQ/noisy_butterflied_diffusion", "UNet2DModel", "DDPMScheduler"),
    ModelRef("Pal10Palash", "Pal10/Palash_gen_butterflies", "UNet2DModel", "DDPMScheduler"),
    # ModelRef("Apocalypse19Ceyda", "Apocalypse-19/ceyda-butterflies-64", "UNet2DModel", "DDPMScheduler"),
    # ModelRef("KsamlMnist", "ksaml/mnist-fashion_64", "UNet2DModel", "DDPMScheduler"),
    # ModelRef("SalehiUnit1", "salehi/salehi_unit1", "UNet2DModel", "DDPMScheduler"),
    # ModelRef("CompVisLDM", "CompVis/ldm-celebahq-256", "UNet2DModel", "DDIMScheduler"),
    # ModelRef("GoogleNcsnpp", "google/ncsnpp-bedroom-256", "UNet2DModel", "ScoreSdeVeScheduler"),
}

# ======================================

for dataset_ref in Datasets:
    for model_ref in Models:
        output_path = Path(str(f'./UnconditionalImageGen/results/{dataset_ref.name}_{dataset_ref.row}/{model_ref.name}/'))
        train(dataset_ref, model_ref, output_path)
