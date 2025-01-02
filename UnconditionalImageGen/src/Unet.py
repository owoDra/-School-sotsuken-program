import sys
sys.path.append("./src/")

from diffusers import UNet2DModel

# 
# @Return (Model, image_size, channels)
# 
def resolve_unet_UNet2DModel(unet_classname, url, image_size, device):
    unet_config = UNet2DModel.load_config(url)
    unet_config["sample_size"] = image_size
    channels = unet_config["out_channels"]
    model = UNet2DModel.from_config(unet_config)
    model.to(device)
    return (model, channels)

# 
# @Return (Model, channels)
# 
def resolve_unet(unet_classname, url, image_size, device):
    match unet_classname:
        case "UNet2DModel":
            return resolve_unet_UNet2DModel(unet_classname, url, image_size, device)
        
    return resolve_unet_UNet2DModel(unet_classname, url, image_size, device)