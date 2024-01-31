import configparser
import os
import torch
# from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image

config = configparser.ConfigParser()
config.read('config/config.ini')

model = config['SDXL']['sdxl_model']
local=config['SDXL']['local']
model_id=config['SDXL']['sdxl_model_id']

if local=="1":
    print("Use Local Model")
    if os.path.isdir(model) is False:
        raise ValueError("Model is not detected.")
else:
    model= model_id
    print("Use Online Model")
    
# load model
# pipe = DiffusionPipeline.from_pretrained(model_sdxl_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")







