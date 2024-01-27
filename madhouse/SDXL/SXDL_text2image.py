from madhouse.SDXL import pipe
import datetime

def SDXL_inference(prompt):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    save_path="outputs/"+"output_"+current_time+".png"
    image = pipe(prompt=prompt).images[0]
    image.save(save_path)
    return save_path

# def SDXL_inference2(prompt):
#     current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
#     save_path="outputs/"+"output_"+current_time+".png"
#     image = pipeline_text2image(prompt=prompt).images[0]
#     image.save(save_path)
#     return save_path