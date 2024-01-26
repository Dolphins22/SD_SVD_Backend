from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    model_sdxl_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

def SDXL_inference(prompt):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    save_path="outputs/"+"output_"+current_time+".png"
    image = pipeline_text2image(prompt=prompt).images[0]
    image.save(save_path)
    return save_path
    


