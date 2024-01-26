import configparser
import os

if os.path.isdir('models/stable-video-diffusion-img2vid-xt') is False:
    os.system("""
                git lfs install;
                git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt models
            """) 

config = configparser.ConfigParser()
config.read('config/config.ini')

model_svd_xt = config['SVD']['svd_img2vid-xt']
repeat = config['SVD']['repeat']
decode_chunk_size = config['SVD']['decode_chunk_size']
motion_bucket_id = config['SVD']['motion_bucket_id']
noise_aug_strength = config['SVD']['noise_aug_strength']
num_frames = config['SVD']['num_frames']








