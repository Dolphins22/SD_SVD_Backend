from madhouse.SD.SD_text2image import SD_inference
import sys

script_name=sys.argv[0]
prompt=sys.argv[1]
SD_inference(prompt)