import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset

from preprocessing.preprocessingInDisk import (
    AGNewsPreprocessing,
    DBpediaPreprocessing,
    ISIC2019Preprocessing,
    MNISTPreprocessing
)

from .imageDataset import ImageDataset
from .textDatasetInMemory import TextDatasetInMemory
from .mnistDataset import MNISTDataset
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
        images, labels, n_classes, n_channels = ISIC2019Preprocessing().load(
            csv_path=os.path.join(data_pth, dataset_name, "ISIC_2019_Training_GroundTruth.csv"),
            img_dir=os.path.join(data_pth, dataset_name, "images")
        )

        full_dataset = ImageDataset(
            images_name=images,
            labels=labels,
            class_num=n_classes,
            n_channels=n_channels,
            img_full_dir=os.path.join(data_pth, dataset_name, "images"),
            transform=transform
        )

    # ---------------- MNIST ----------------
    elif dataset_name == "MNIST":
        images, labels, n_classes, n_channels = MNISTPreprocessing().load(
            path=data_pth,
            transform=transform
        )

        full_dataset = MNISTDataset(
            images=images,
            labels=labels,
            class_num=n_classes,
            n_channels=n_channels
        )

    # ---------------- TEXT ----------------
    elif dataset_name == "AGNews":
        encoded_texts, attention_masks, labels, num_classes, _ = AGNewsPreprocessing().load(
            csv_train_path=os.path.join(data_pth, dataset_name, "train.csv"),
            csv_test_path=os.path.join(data_pth, dataset_name, "test.csv")
        )

        full_dataset = TextDatasetInMemory(
            encoded_texts=encoded_texts,
            attention_masks=attention_masks,
            labels=labels,
            class_num=num_classes
        )

    elif dataset_name == "DBpedia":
        encoded_texts, attention_masks, labels, num_classes, _ = DBpediaPreprocessing().load(
            csv_train_path=os.path.join(data_pth, dataset_name, "train.csv"),
            csv_test_path=os.path.join(data_pth, dataset_name, "test.csv")
        )

        full_dataset = TextDatasetInMemory(
            encoded_texts=encoded_texts,
            attention_masks=attention_masks,
            labels=labels,
            class_num=num_classes
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ---------------- SPLIT ----------------
    train_idx, val_idx, test_idx = stratified_split(
        labels=labels,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )

    train_dataset = IndexedDataset(full_dataset, train_idx)
    val_dataset = IndexedDataset(full_dataset, val_idx)
    test_dataset = IndexedDataset(full_dataset, test_idx)

    meta = {
        "class_num": n_classes if dataset_name in ["ISIC2019", "MNIST"] else num_classes,
        "labels": labels,
        "n_channels": n_channels
    }

    return train_dataset, val_dataset, test_dataset, meta