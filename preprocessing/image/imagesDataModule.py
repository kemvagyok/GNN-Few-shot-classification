import os
import torch
import tqdm
from PIL import Image

class ImagesDataModule:
    def __init__(self, path_raw, transform):
        self.path = path_raw
        self.img_dir = os.path.join(path_raw, "images")
        self.transform = transform

    def load_images(self, paths, labels):
        imgs = []
        for p in tqdm.tqdm(paths):
            img = Image.open(p)
            imgs.append(self.transform(img))

        return torch.stack(imgs), torch.tensor(labels)
