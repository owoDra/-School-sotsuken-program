import sys
sys.path.append("./src/")

from diffusers import DDPMPipeline

# 
# @Return Pipeline
# 
def resolve_pipeline_DDPM(model, scheduler):
    return DDPMPipeline(unet = model, scheduler = scheduler)

# 
# @Return Pipeline
# 
def resolve_pipeline(scheduler_classname, model, scheduler):
    match scheduler_classname:
        case "DDPMScheduler":
            return resolve_pipeline_DDPM(model, scheduler)
        
    return resolve_pipeline_DDPM(model, scheduler)
