from madhouse.SD import pipe
import datetime

def SD_inference(prompt):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    save_path="outputs/"+"output_"+current_time+".png"
    image = pipe(prompt=prompt).images[0]
    image.save(save_path)
    return save_path


