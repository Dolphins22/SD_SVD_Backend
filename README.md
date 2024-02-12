# SD_backend
This is a Stable Diffusion backend for running SD1.5, SDXL and SVD models. 

### Python venv
```
python -m venv .venv_sd
source .venv_sd/bin/activate
```

#### Prerequisite:<br>
1, Install torch<br>
2, Install requirements.txt

Huggingface Login
`huggingface-cli login`

Set Python sys.path<br>
`export PYTHONPATH=$PYTHONPATH:/home/ubuntu/SD_backend`

#### Run<br>
`cd /home/ubuntu/SD_backend`

`python scripts/SD_run.py "An image of a bunny in Picasso style"`

`python scripts/SDXL_run.py "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"`

`python scripts/SVD_run.py "/home/ubuntu/SD_backend/inputs/Example.png"`
