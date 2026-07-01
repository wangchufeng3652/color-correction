import os
import torch
import cv2
import numpy as np


# =====================
# root
# =====================

ROOT = r"C:\Users\HP\Desktop\second\luf-net"##path

os.chdir(ROOT)



from models.unet_film_lab import LiteUNetFiLMResidualLab

from utils.color import rgb_to_lab, lab_to_rgb




# =====================
# Path
# =====================


image_path = r"data/example/inputs/00001_input.jpg"


save_path = r"outputs/00001_prediction.jpg"


model_path = r"weights/best_model.pth"




device=torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"

)




# =====================
# Read image
# =====================


if not os.path.exists(image_path):

    raise FileNotFoundError(
        image_path
    )



img=cv2.imread(

    image_path

)



if img is None:

    raise ValueError(
        "Image load failed"
    )




img=cv2.cvtColor(

    img,

    cv2.COLOR_BGR2RGB

)




# H W C
# ->
# C H W


img=torch.tensor(

    img,

    dtype=torch.float32

).permute(2,0,1)/255.0




# RGB -> Lab

img_lab=rgb_to_lab(

    img

)



# add batch

img_lab=img_lab.unsqueeze(0)



img_lab=img_lab.to(device)





# =====================
# Camera vector
# 和 vectors.json 保持一致
# =====================


vector=torch.tensor(

    [

        1.8,       # aperture

        0.001,     # exposure_time

        100,       # iso

        0.35,      # band_G

        0.25,      # band_R

        0.45,      # band_NIR

        0.38       # band_RE


    ],

    dtype=torch.float32

)



vector=vector.unsqueeze(0)



vector=vector.to(device)





# =====================
# Load model
# =====================


model=LiteUNetFiLMResidualLab(

    vector_dim=7

).to(device)



checkpoint=torch.load(

    model_path,

    map_location=device

)



model.load_state_dict(

    checkpoint

)



model.eval()



print("Model loaded")





# =====================
# Predict
# =====================


with torch.no_grad():


    pred=model(

        img_lab,

        vector

    )




# =====================
# Lab -> RGB
# =====================


pred_rgb=lab_to_rgb(

    pred.squeeze(0).cpu()

)




# C H W
# ->
# H W C


out=(

    pred_rgb

    .permute(1,2,0)

    .numpy()

    *255

).astype(np.uint8)





# RGB -> BGR

out=cv2.cvtColor(

    out,

    cv2.COLOR_RGB2BGR

)




# create folder

os.makedirs(

    os.path.dirname(save_path),

    exist_ok=True

)



cv2.imwrite(

    save_path,

    out

)



print(

    "Saved:",

    save_path

)