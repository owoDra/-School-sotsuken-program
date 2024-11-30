import sys
sys.path.append("./src/")

from diffusers import DiffusionPipeline

# 
# @Return (UnetClassName, SchedulerClassName)
# 
def get_ref_class_names_from_model(url):
    pipeline = DiffusionPipeline.from_pretrained(url)
    return (pipeline.config['unet'][1], pipeline.config['scheduler'][1])
