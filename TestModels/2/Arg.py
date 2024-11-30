_NAME_OR_PATH_DATASET_ = "jiovine/pixel-art-nouns-2k"
_NAME_OR_PATH_MODEL_ = "google/ddpm-cifar10-32"

adam_beta1 = 0.95
adam_beta2 = 0.999
adam_weight_decay = 1e-6
adam_epsilon = 1e-08
lr_rate = 1e-4
lr_scheduler = "cosine"
lr_warmup_steps = 100
gradient_accumulation_steps = 1
num_epochs = 10
save_model_epochs = 10
prediction_type = "epsilon"
ddpm_num_inference_steps = 1000
dataloader_num_workers = 0
train_batch_size = 16
output_dir = "./trained_models/"

eval_batch_size = 5
result_dir = "./trained_result/"