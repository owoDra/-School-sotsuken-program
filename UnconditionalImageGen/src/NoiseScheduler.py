import sys
sys.path.append("./src/")

from diffusers import DDPMScheduler
from diffusers import DDIMScheduler
from diffusers import ScoreSdeVeScheduler

# 
# @Return Scheduler
# 
def resolve_noise_scheduler_DDPM(unet_classname, url):
    noise_scheduler_config = DDPMScheduler.load_config(url)
    return DDPMScheduler.from_config(noise_scheduler_config)

# 
# @Return Scheduler
# 
def resolve_noise_scheduler_DDIM(unet_classname, url):
    noise_scheduler_config = DDIMScheduler.load_config(url)
    return DDIMScheduler.from_config(noise_scheduler_config)

# 
# @Return Scheduler
# 
def resolve_noise_scheduler_SSV(unet_classname, url):
    noise_scheduler_config = ScoreSdeVeScheduler.load_config(url)
    return ScoreSdeVeScheduler.from_config(noise_scheduler_config)

# 
# @Return Scheduler
# 
def resolve_noise_scheduler(scheduler_classname, url):
    match scheduler_classname:
        case "DDPMScheduler":
            return resolve_noise_scheduler_DDPM(scheduler_classname, url)
        case "DDIMScheduler":
            return resolve_noise_scheduler_DDIM(scheduler_classname, url)
        case "ScoreSdeVeScheduler":
            return resolve_noise_scheduler_SSV(scheduler_classname, url)
        
    return resolve_noise_scheduler_DDPM(scheduler_classname, url)