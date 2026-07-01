import torch
import time
from torch.utils.data import DataLoader
import os

ROOT = r"C:\Users\HP\Desktop\second\luf-net"##path

os.chdir(ROOT)


from models.unet_film_lab import LiteUNetFiLMResidualLab
from utils.dataset import ImageVectorDatasetLab



# =====================
# Configuration
# =====================

device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



INPUT_DIR="data/example/inputs"

TARGET_DIR="data/example/targets"

VECTOR_JSON="data/example/vectors.json"


MODEL_PATH="weights/best_model.pth"



BATCH_SIZE=64



# =====================
# Dataset
# =====================


dataset=ImageVectorDatasetLab(

    INPUT_DIR,

    TARGET_DIR,

    VECTOR_JSON,

    device

)



loader=DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)



# =====================
# Model
# =====================


model=LiteUNetFiLMResidualLab().to(device)


model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device
    )

)


model.eval()



# =====================
# Evaluation
# =====================


color_error=0

mape_total=0



start=time.time()



with torch.no_grad():


    for imgs,vecs,targets in loader:



        pred=model(

            imgs,

            vecs

        )



        # Lab color distance

        diff=pred-targets



        delta_e=torch.sqrt(

            torch.sum(

                diff**2,

                dim=1

            )

        )



        color_error += (

            delta_e.mean().item()

            *

            imgs.size(0)

        )



        mape=torch.mean(

            torch.abs(pred-targets)

            /

            (torch.abs(targets)+1e-6)

        )*100



        mape_total += (

            mape.item()

            *

            imgs.size(0)

        )




elapsed=time.time()-start



color_error /= len(dataset)

mape_total /= len(dataset)



fps=len(dataset)/elapsed



# =====================
# Save result
# =====================


result=f"""

Maize Evaluation Result

Color Error (Lab):
{color_error*255:.4f}


MAPE:
{mape_total:.4f}%


Total Time:
{elapsed:.3f}s


FPS:
{fps:.2f}

"""


print(result)



with open(

"outputs/evaluation_results.txt",

"w"

) as f:

    f.write(result)
