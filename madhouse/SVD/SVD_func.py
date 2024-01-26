import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from random import randint
import datetime   
import os

# load model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_svd_xt, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

def SVD_inference(image_path):
    image = load_image(image_path)
    image = image.resize((1024, 576)) # image input must by 1024*576 
    name_without_extension = os.path.splitext(image_path)[0]
    
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
        return save_path

        
