import torch
import torchvision
import cv2
import numpy as np

# MEAN = [0.4438, 0.4440, 0.4438]
# STD  = [0.2530, 0.2529, 0.2530]
MEAN = [0.4439, 0.4439, 0.4439]
STD  = [0.2529, 0.2529, 0.2529]

def get_normalize():
    return torchvision.transforms.Normalize(mean=MEAN, std=STD)


def get_normalize_transform():
    return (
        torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        get_normalize(),
    ]))


def get_pretreatment_transform():
    return (
        torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ]))


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_grid_size)

    def __call__(self, img):
        """
        img: a PIL Image OR a torch Tensor (C,H,W) with values in [0,1] or [0,255]
        """
        # Convert PIL → NumPy
        if not isinstance(img, torch.Tensor):
            img = np.array(img)

        # Handle RGB
        # Convert RGB → LAB (CLAHE works best on L channel)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE only on L channel
        l2 = self.clahe.apply(l)

        # Recombine and convert back LAB → RGB
        lab_clahe = cv2.merge((l2, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        # Convert NumPy → Tensor
        return torchvision.transforms.ToPILImage()(img_clahe)