import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, INFO, Evaluator
from preprocessing import _build_transform, fileLoading_ChestX

 
'''
Filtering the training dataset to have at most max_size_as_class samples per class.
'''
def traindatasetFiltering(targets, num_class, max_size_as_class):
    indexs = np.arange(targets.shape[0])
    
    train_targets_index_bool = [targets == y for y in range(num_class)]# Choosing X images from each classes as labeled
    
    train_targets_indexs = [np.asarray(indexs[index_bool]) for index_bool in train_targets_index_bool]
    
    for c in range(num_class):
        np.random.shuffle(train_targets_indexs[c]) # Randomly shuffle the indices of each class to ensure random sampling
        assert len(train_targets_indexs[c]) >= max_size_as_class, \
          f"Class {c} has only {len(train_targets_indexs[c])} samples, but max_size_as_class={max_size_as_class}"
    
    train_mask_index = np.hstack(np.array([train_targets_indexs[target][:max_size_as_class] for target in range(num_class)])) #Labeling
        
    return train_mask_index


def dataLoading_MNIST():
    grayscale = True
    transform = _build_transform(28, grayscale)
    
    train_dataset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)
    test_dataset = datasets.MNIST('./data',
                                train=False,
                                download=True,
                                transform=transform)
    
    n_classes = len(train_dataset.classes)
    n_channels = 1
    
    #--- Train
    train_x = train_dataset.data.unsqueeze(1) / 255
    train_y = train_dataset.targets
    #--- Test
    test_x = test_dataset.data.unsqueeze(1) / 255
    test_y = test_dataset.targets
    
    return (train_x, train_y, test_x, test_y, n_classes, n_channels)


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

def dataLoading_ChestX(
        ptFile = "data/nih_chest_xray/ChestX_data.pt", 
        force_reload=False,
        **kwargs
        ):
    
    grayscale = True
    transform = _build_transform(28, grayscale)

    if os.path.exists(ptFile) and not force_reload:
        data = torch.load(ptFile)
        return (
            data["train_x"],
            data["train_y"],
            data["test_x"],
            data["test_y"],
            data["n_classes"],
            data["n_channels"]
        )
    
    train_x, train_y, test_x, test_y, n_classes, n_channels = fileLoading_ChestX(
        transform=transform,
        **kwargs
    )

    torch.save({
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "n_classes": n_classes,
        "n_channels": n_channels
    }, ptFile)

    return train_x, train_y, test_x, test_y, n_classes, n_channels


class FewShotDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
     return self.x[idx], self.y[idx]



if __name__ == "__main__":
    dataLoading_ChestX()
    
