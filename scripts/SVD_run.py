from madhouse.SVD.SVD_func import SVD_inference
import sys

# (576x1024) videos only
script_name=sys.argv[0]
image_path=sys.argv[1]
SVD_inference(image_path)
