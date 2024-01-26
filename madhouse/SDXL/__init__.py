import configparser
import os

if os.path.isdir('models/stable-diffusion-xl-base-1.0') is False:
    os.system("""
                git lfs install;
                git clone git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 models
            """) 

config = configparser.ConfigParser()
config.read('config/config.ini')

model_sdxl_model = config['SDXL']['sdxl_model_base_1.0 ']








