import torch
import cv2
import numpy as np


def rgb_to_lab(img_tensor):
    """
    RGB -> Lab
    Input:
        C×H×W or B×C×H×W
    Output:
        Lab tensor normalized to [0,1]
    """
    if img_tensor.dim() == 3:
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab = torch.tensor(lab, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return lab.to(img_tensor.device)

    elif img_tensor.dim() == 4:
        return torch.stack([rgb_to_lab(img_tensor[i]) for i in range(img_tensor.size(0))])


def lab_to_rgb(lab_tensor):
    """
    Lab -> RGB
    Input:
        C×H×W or B×C×H×W
    Output:
        RGB tensor normalized to [0,1]
    """
    if lab_tensor.dim() == 3:
        lab = (lab_tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return rgb.to(lab_tensor.device)

    elif lab_tensor.dim() == 4:
        return torch.stack([lab_to_rgb(lab_tensor[i]) for i in range(lab_tensor.size(0))])
