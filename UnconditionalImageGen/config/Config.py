def Config():
    def __init__(self):
        self.adam_beta1 = 0.95
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-6
        self.adam_epsilon = 1e-08
        self.lr_rate = 1e-4
        self.lr_scheduler = "cosine"
        self.lr_warmup_steps = 100
        self.gradient_accumulation_steps = 1
        self.num_epochs = 10
        self.save_model_epochs = 10
        self.prediction_type = "epsilon"
        self.ddpm_num_inference_steps = 1000
        self.dataloader_num_workers = 0
        self.train_batch_size = 16
        

_NAME_OR_PATH_DATASET_ = "jiovine/pixel-art-nouns-2k"
_NAME_OR_PATH_MODEL_ = "google/ddpm-cifar10-32"


output_dir = "./trained_models/"

eval_batch_size = 5
result_dir = "./trained_result/"

