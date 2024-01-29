from madhouse.SVD import pipe
import torch
from diffusers.utils import load_image, export_to_video
from random import randint
import configparser
import datetime   
import os

config = configparser.ConfigParser()
config.read('config/config.ini')

repeat = config.getint('SVD', 'repeat')
decode_chunk_size = config.getint('SVD', 'decode_chunk_size')
motion_bucket_id = config.getint('SVD', 'motion_bucket_id')
noise_aug_strength = config.getfloat('SVD', 'noise_aug_strength')
num_frames = config.getint('SVD', 'num_frames')

def SVD_inference(image_path):
    image = load_image(image_path)
    image = image.resize((1024, 576))
    name_without_extension = os.path.splitext(image_path)[0].split('/')[-1]
    
    for i in range(repeat):
        # change seed each time repeat
        seed_num=randint(1, 100)
        generator = torch.manual_seed(seed_num)
        
        # inference
        frames = pipe(
                        image, \
                        decode_chunk_size=decode_chunk_size, \
                        generator=generator, 
                        motion_bucket_id=motion_bucket_id, \
                        noise_aug_strength=noise_aug_strength, \
                        num_frames=num_frames
                    ).frames[0]
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        save_path="outputs/"+name_without_extension+"_output_"+str(i)+"_"+current_time+".mp4"
        export_to_video(frames, save_path, fps=7)
        print(save_path)

        
