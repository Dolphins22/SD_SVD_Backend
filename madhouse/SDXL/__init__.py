import configparser
import os
import torch
# from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image

config = configparser.ConfigParser()
config.read('config/config.ini')

model_sdxl_model = config['SDXL']['sdxl_model']

if os.path.isdir(model_sdxl_model) is False:
    os.system("""
                git lfs install;
                git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 models
            """) 

# load model
# pipe = DiffusionPipeline.from_pretrained(model_sdxl_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
pipe = AutoPipelineForText2Image.from_pretrained(model_sdxl_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")







