import os
import json
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image

from .color import rgb_to_lab



class ImageVectorDatasetLab(Dataset):

    def __init__(
        self,
        input_dir,
        target_dir,
        json_file,
        device
    ):

        self.device=device


        self.inputs=[
        os.path.join(input_dir,x)
        for x in os.listdir(input_dir)
        ]


        self.targets=[
        os.path.join(target_dir,x)
        for x in os.listdir(target_dir)
        ]


        with open(json_file) as f:

            self.vec=json.load(f)



    def __len__(self):

        return len(self.inputs)



    def __getitem__(self,i):


        x=read_image(
            self.inputs[i]
        ).float()/255


        y=read_image(
            self.targets[i]
        ).float()/255



        v=torch.tensor(
            list(self.vec[str(i)].values()),
            dtype=torch.float32
        )


        return (

            rgb_to_lab(x).to(self.device),

            v.to(self.device),

            rgb_to_lab(y).to(self.device)

        )
