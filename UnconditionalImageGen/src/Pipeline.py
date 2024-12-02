import sys
sys.path.append("./src/")

from diffusers import DDPMPipeline
from diffusers import DDIMPipeline
from diffusers import ScoreSdeVePipeline

# 
# @Return Pipeline
# 
def resolve_pipeline_DDPM(model, scheduler):
    return DDPMPipeline(unet = model, scheduler = scheduler)

# 
# @Return Pipeline
# 
def resolve_pipeline_DDIM(model, scheduler):
    return DDIMPipeline(unet = model, scheduler = scheduler)

# 
# @Return Pipeline
# 
def resolve_pipeline_SSV(model, scheduler):
    return ScoreSdeVePipeline(unet = model, scheduler = scheduler)

# 
# @Return Pipeline
# 
def resolve_pipeline(scheduler_classname, model, scheduler):
    match scheduler_classname:
        case "DDPMScheduler":
            return resolve_pipeline_DDPM(model, scheduler)
        case "DDIMScheduler":
            return resolve_pipeline_DDIM(model, scheduler)
        case "ScoreSdeVeScheduler":
            return resolve_pipeline_SSV(model, scheduler)
        
    return resolve_pipeline_DDPM(model, scheduler)
