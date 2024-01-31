import configparser
import os
import torch
from diffusers import StableVideoDiffusionPipeline

config = configparser.ConfigParser()
config.read('config/config.ini')

model = config['SVD']['svd_img2vid-xt']
local=config['SVD']['local']
model_id=config['SVD']['svd_img2vid-xt_id']

if local=="1":
    print("Use Local Model")
    if os.path.isdir(model) is False:
        raise ValueError("Model is not detected.")
else:
    model= model_id
    print("Use Online Model")

# load model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()









