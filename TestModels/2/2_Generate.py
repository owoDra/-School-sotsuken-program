import sys
sys.path.append("./src/")

from PIL import Image

import torch
from torchvision.utils import save_image

from diffusers import DDPMPipeline

import Arg


# --------------------------------------------
# モデル読み込み
# --------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = DDPMPipeline.from_pretrained(Arg.output_dir)
pipeline.to(device)

# --------------------------------------------
# 画像生成
# --------------------------------------------

generator = torch.Generator(device=pipeline.device).manual_seed(0)

images = pipeline(
    generator=generator,
    batch_size=Arg.eval_batch_size,
    num_inference_steps=Arg.ddpm_num_inference_steps,
    output_type="pil",
).images


# --------------------------------------------
# 画像保存
# --------------------------------------------

for idx, image in enumerate(images):
    image.save(str(f'{Arg.result_dir}/sample-{idx}.png'))
