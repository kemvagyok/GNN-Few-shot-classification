# =========================================================
# STANDARD LIBRARIES
# =========================================================
import os

# =========================================================
# THIRD-PARTY LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# =========================================================
# PROJECT MODULES
# =========================================================
from .preprocessing import build_transform, ISIC2019Dataset, ChestXDataset

# =========================================================
# UTILITIES
# =========================================================
from collections import Counter

def get_class_distribution(targets, num_classes=None):
    """
    targets: torch.Tensor vagy numpy array (N,)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    counter = Counter(targets)

    if num_classes is not None:
        return {i: counter.get(i, 0) for i in range(num_classes)}

    return dict(counter)

def get_multilabel_distribution(targets):
    """
    targets: (N, C) multi-hot tensor
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return targets.sum(axis=0)

def print_distribution(dist, title="Distribution"):
    print(f"\n--- {title} ---")
    total = sum(dist.values()) if isinstance(dist, dict) else dist.sum()

    if isinstance(dist, dict):
        for k, v in dist.items():
            print(f"Class {k}: {v} ({v/total:.2%})")
    else:
        for i, v in enumerate(dist):
            print(f"Class {i}: {v} ({v/total:.2%})")

'''
Filtering the training dataset to have at most max_size_as_class samples per class.
'''
def traindatasetFiltering(targets, num_class, max_size_as_class):
    indexs = np.arange(targets.shape[0])
    
    train_targets_index_bool = [targets == y for y in range(num_class)]# Choosing X images from each classes as labeled
    
    train_targets_indexs = [np.asarray(indexs[index_bool]) for index_bool in train_targets_index_bool]
    
    for c in range(num_class):
        np.random.shuffle(train_targets_indexs[c]) # Randomly shuffle the indices of each class to ensure random sampling
        #assert len(train_targets_indexs[c]) >= max_size_as_class, \
        if len(train_targets_indexs[c]) >= max_size_as_class:
            print(f"Class {c} has only {len(train_targets_indexs[c])} samples, but max_size_as_class={max_size_as_class}")
    
    train_mask_index = np.hstack(
        [train_targets_indexs[target][:max_size_as_class] for target in range(num_class)]
    )

    return train_mask_index

def dataLoading_ISIC2019(
        data_pth,
        files_size=4000,
        img_size=28,
        force_reload=False,
        **kwargs):

        ptFile = os.path.join(data_pth, f"preprocessed/ISIC2019_data_{files_size}_{img_size}.pt")
        if os.path.exists(ptFile) and not force_reload:
            print("Loading cached ISIC2019 data...")
            data = torch.load(ptFile)

            train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels = (
                data["train_x"],
                data["train_y"],
                data["val_x"],
                data["val_y"],
                data["test_x"],
                data["test_y"],
                data["n_classes"],
                data["n_channels"]
            )
            train_dist = get_class_distribution(train_y, n_classes)
            test_dist = get_class_distribution(test_y, n_classes)
            print_distribution(train_dist, "ISIC2019 Train")
            print_distribution(test_dist, "ISIC2019 Test")
            return  train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels
  
        print("Not found cached ISIC2019 data, loading from source... this may take a while.")

        train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels = ISIC2019Dataset(
            img_dir=os.path.join(data_pth, "raw/ISIC2019/images"), 
            img_size=img_size, 
            **kwargs).load(files_size,files_size)

        torch.save({
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y,
            "n_classes": n_classes,
            "n_channels": n_channels
        }, ptFile)

        train_dist = get_class_distribution(train_y, n_classes)
        test_dist = get_class_distribution(test_y, n_classes)

        print_distribution(train_dist, "ISIC2019 Train")
        print_distribution(test_dist, "ISIC2019 Test")

        return train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels
        


def dataLoading_ChestX(
        data_pth, 
        files_size=4000,
        img_size = 28,
        force_reload=False,
        **kwargs
        ):
    
    ptFile = os.path.join(data_pth, f"preprocessed/ChestX_data_{files_size}_{img_size}.pt")

    if os.path.exists(ptFile) and not force_reload:
        print("Loading cached ChestX data...")
        data = torch.load(ptFile)
        return (
            data["train_x"],
            data["train_y"],
            data["val_x"],
            data["val_y"],
            data["test_x"],
            data["test_y"],
            data["n_classes"],
            data["n_channels"]
        )
    
    print("Not found cached ChestX data, loading from source... this may take a while.")
    
    train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels = ChestXDataset(
        img_dir=os.path.join(data_pth, "ChestX/images"),
        csv_path=os.path.join(data_pth, "ChestX/Data_Entry_2017.csv"),
        train_index_path=os.path.join(data_pth, "ChestX/train_val_list.txt"),
        test_index_path=os.path.join(data_pth, "ChestX/test_list.txt"),
        img_size=img_size,
        **kwargs
    ).load(files_size,files_size)

    torch.save({
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "n_classes": n_classes,
        "n_channels": n_channels
    }, ptFile)

    train_dist = get_class_distribution(train_y, n_classes)
    test_dist = get_class_distribution(test_y, n_classes)

    print_distribution(train_dist, "ChestX Train")
    print_distribution(test_dist, "ChestX Test")

    return train_x, train_y, test_x, test_y, n_classes, n_channels



def dataLoading_MNIST(data_pth,val_ratio=0.1):
    grayscale = True
    transform = build_transform(28, grayscale)
    data_pth = os.path.join(data_pth, "raw", "MNIST", "raw")
    # --- MNIST betöltés
    train_dataset = datasets.MNIST(data_pth,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(data_pth,
                                  train=False,
                                  download=True,
                                  transform=transform)
    
    n_classes = len(train_dataset.classes)
    n_channels = 1

    # --- Train tensorok
    train_x = train_dataset.data.unsqueeze(1) / 255.0
    train_y = train_dataset.targets

    # --- Test tensorok
    test_x = test_dataset.data.unsqueeze(1) / 255.0
    test_y = test_dataset.targets

    # --- Train --> Train + Val split (tensor alapú, nem Subset!)
    num_train = len(train_x)
    num_val = int(num_train * val_ratio)

    # Reprodukálható shuffle
    perm = torch.randperm(num_train)

    val_idx = perm[:num_val]
    train_idx = perm[num_val:]

    # Train split
    train_x_split = train_x[train_idx]
    train_y_split = train_y[train_idx]

    # Val split
    val_x = train_x[val_idx]
    val_y = train_y[val_idx]

    # --- Print distributions
    print_distribution(get_class_distribution(train_y_split, n_classes), "MNIST Train")
    print_distribution(get_class_distribution(val_y,           n_classes), "MNIST Val")
    print_distribution(get_class_distribution(test_y,          n_classes), "MNIST Test")

    return (train_x_split, train_y_split,
            val_x, val_y,
            test_x, test_y,
            n_classes, n_channels)

'''
def dataLoading_medMNIST():
    data_flag = 'dermamnist'
    download = True
    
    info = INFO[data_flag]
    
    n_classes = len(info['label'])
    n_channels = info['n_channels']
    
    
    grayscale = True
    transform = _build_transform(28, grayscale)
    
    train_dataset = DermaMNIST(split="train",
                             download=True,
                             size = 28,
                             transform=transform)
    test_dataset = DermaMNIST(split="test",
                            download=True,
                            size = 28,
                            transform=transform)
    #------ train
    train_x = train_dataset.imgs
    train_x = torch.asarray(train_x, dtype = torch.float32).permute(0,3,1,2) / 255
    
    train_y = train_dataset.labels
    train_y = train_y.reshape(len(train_y))
    train_y = torch.asarray(train_y)
    
    #------ test
    test_x = test_dataset.imgs
    test_x = torch.asarray(test_x, dtype = torch.float32).permute(0,3,1,2) / 255
    
    test_y = test_dataset.labels
    test_y = test_y.reshape(len(test_y))
    test_y = torch.asarray(test_y)
    
    return (train_x, train_y, test_x, test_y, n_classes, n_channels)
'''



class FewShotDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
     return self.x[idx], self.y[idx]    
