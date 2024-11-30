import sys
sys.path.append("./src/")

from diffusers import UNet2DModel

# 
# @Return (Model, image_size, channels)
# 
def resolve_unet_UNet2DModel(unet_classname, url, device):
    unet_config = UNet2DModel.load_config(url)
    image_size = unet_config["sample_size"]
    channels = unet_config["out_channels"]
    model = UNet2DModel.from_config(unet_config)
    model.to(device)
    return (model, image_size, channels)

# 
# @Return (Model, image_size, channels)
# 
def resolve_unet(unet_classname, url, device):
    match unet_classname:
        case "UNet2DModel":
            return resolve_unet_UNet2DModel(unet_classname, url, device)
        
    return resolve_unet_UNet2DModel(unet_classname, url, device)