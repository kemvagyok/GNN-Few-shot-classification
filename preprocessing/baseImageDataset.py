import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

def build_transform(img_size=28, grayscale=True):

    mean, std = ([0.5], [0.5]) if grayscale else (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
    
    transform_list = []

    transform_list.append(transforms.Resize((img_size)))
    transform_list.append(transforms.CenterCrop((img_size)))

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)

class BaseImageDataset:
    def __init__(self, path_raw, img_size=28, grayscale=True):
        self.path = path_raw
        self.img_dir = os.path.join(path_raw, "images")
        self.img_size = img_size
        self.grayscale = grayscale
        self.transform = build_transform(img_size=img_size, grayscale=grayscale)

    def load_images(self, paths, labels):
        imgs = []
        for p in paths:
            img = Image.open(p).convert("L" if self.grayscale else "RGB")
            imgs.append(self.transform(img))

        return torch.stack(imgs), torch.tensor(labels)

    def split_data(self, paths, labels, test_size=0.2, val_size=0.2):
        train_p, test_p, train_y, test_y = train_test_split(
            paths, labels, stratify=labels, test_size=test_size, random_state=42
        )

        train_p, val_p, train_y, val_y = train_test_split(
            train_p, train_y, test_size=val_size, random_state=42
        )

        return train_p, val_p, test_p, train_y, val_y, test_y