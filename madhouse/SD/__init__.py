import configparser
import os
from diffusers import StableDiffusionPipeline
import torch

config = configparser.ConfigParser()
config.read('config/config.ini')

model = config['SD']['sd_model']
local=config['SD']['local']
model_id=config['SD']['sd_model_id']

if local==1 and os.path.isdir(model) is False:
    raise ValueError("Model is not detected.")
else:
    model= model_id

# load model
# pipe = DiffusionPipeline.from_pretrained(model_sdxl_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipe = pipe.to("cuda")






