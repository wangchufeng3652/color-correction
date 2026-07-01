# Color correction

LUF-Net: Physically informed color calibration for UAV RGB images.

## Installation

pip install -r requirements.txt


## Dataset

data/example/

input:
UAV RGB images

target:
reference corrected images

vectors.json:
camera exposure and spectral parameters


## Training

python train.py


## Evaluation

python eval.py


## Inference

python inference.py


## Model

Pretrained weights:

weights/best_model.pth


## Reproducibility

This repository provides:

- source code
- pretrained model
- example dataset
- evaluation scripts
- inference scripts
- documentation
