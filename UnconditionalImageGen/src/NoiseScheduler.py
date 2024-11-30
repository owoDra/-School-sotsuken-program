import sys
sys.path.append("./src/")

from diffusers import DDPMScheduler

# 
# @Return Scheduler
# 
def resolve_noise_scheduler_DDPM(unet_classname, url):
    noise_scheduler_config = DDPMScheduler.load_config(url)
    return DDPMScheduler.from_config(noise_scheduler_config)

# 
# @Return Scheduler
# 
def resolve_noise_scheduler(scheduler_classname, url):
    match scheduler_classname:
        case "DDPMScheduler":
            return resolve_noise_scheduler_DDPM(scheduler_classname, url)
        
    return resolve_noise_scheduler_DDPM(scheduler_classname, url)