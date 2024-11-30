import sys
sys.path.append("./src/")

from diffusers.optimization import get_scheduler

# 
# @Return lr Scheduler
# 
def get_lr_scheduler(optimizer, num_training_steps, num_warmup_steps = 100, type = "cosine"):
    return get_scheduler(
            type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
