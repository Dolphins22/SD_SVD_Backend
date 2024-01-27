from madhouse.SDXL.SXDL_text2image import SDXL_inference
import sys

script_name=sys.argv[0]
prompt=sys.argv[1]
SDXL_inference(prompt)