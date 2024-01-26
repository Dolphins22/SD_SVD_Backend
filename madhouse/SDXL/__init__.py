import configparser
import os

config = configparser.ConfigParser()
config.read('config/config.ini')

model_sdxl_model = config['SDXL']['sdxl_model']

if os.path.isdir(model_sdxl_model) is False:
    os.system("""
                git lfs install;
                git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 models
            """) 







