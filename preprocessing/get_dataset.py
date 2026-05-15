import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from preprocessing.preprocessingInDisk import (
    ISIC2019Preprocessing,
)

from .image.imageDataset import ImageDataset
from .indexedDataset import IndexedDataset

def stratified_split(labels, train_size, val_size, test_size, seed=42):
    indices = np.arange(len(labels))
    labels = np.array(labels)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=labels
    )

    temp_labels = labels[temp_idx]

    val_ratio = val_size / (val_size + test_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        random_state=seed,
        stratify=temp_labels
    )

    return train_idx, val_idx, test_idx

def get_dataset(
    data_pth,
    dataset_name,
    img_dir,
    transform=None,
    train_size=0.7,
    val_size=0.1,
    test_size=0.2
):

    # ---------------- ISIC ----------------
    if dataset_name == "ISIC2019":
        images_name, labels, n_classes, n_channels = ISIC2019Preprocessing().load(
            csv_path=os.path.join(data_pth, "raw",dataset_name, "ISIC_2019_Training_GroundTruth.csv"),
            img_dir=os.path.join(data_pth,"raw", dataset_name, "images")
        )

        full_dataset = ImageDataset(
            images_name=images_name,
            dataset_path=data_pth,
            dataset_name=dataset_name,
            labels=labels,
            class_num=n_classes,
            n_channels=n_channels,
            img_full_dir=os.path.join(data_pth, "raw", dataset_name, "images"),
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name} or this dataset is loaded from only memory (cached).")

    # ---------------- SPLIT ----------------
    train_idx, val_idx, test_idx = stratified_split(
        labels=labels,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )

    input_size = -1 # Csak a tabuláris dataseteknél lesz értelme, a többi esetben a modell fogja kezelni a bemenet méretét
    train_dataset = IndexedDataset(full_dataset, train_idx)
    val_dataset = IndexedDataset(full_dataset, val_idx)
    test_dataset = IndexedDataset(full_dataset, test_idx)

    meta = {
        "class_num": n_classes,
        "labels": labels,
        "n_channels": n_channels,
        "input_size": input_size
    }

    return train_dataset, val_dataset, test_dataset, meta