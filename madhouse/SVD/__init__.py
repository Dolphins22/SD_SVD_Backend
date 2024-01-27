import configparser
import os
import torch
from diffusers import StableVideoDiffusionPipeline

config = configparser.ConfigParser()
config.read('config/config.ini')

model_svd_xt = config['SVD']['svd_img2vid-xt']

if os.path.isdir(model_svd_xt) is False:
    os.system("""
                git lfs install;
                git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt models
            """) 

# load model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_svd_xt, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()









