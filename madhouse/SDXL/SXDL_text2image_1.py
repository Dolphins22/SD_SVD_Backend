from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(model_sdxl_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

def SDXL_inference(prompt):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    save_path="outputs/"+"output_"+current_time+".png"
    image = pipe(prompt=prompt).images[0]
    image.save(save_path)
    return save_path
    