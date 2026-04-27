import os
import torch
from .preprocessing import get_dataset_class
from .imageDatasetInMemory import ImageDatasetInMemory
from .textDatasetInMemory import TextDatasetInMemory
from .indexedDataset import IndexedDataset
from sklearn.model_selection import train_test_split
import numpy as np


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


# =========================================================
# GENERIC LOADER
# =========================================================
def get_datasetInMemory(
    dataset_name,
    data_pth,
    files_size=4000,
    img_size=28,
    force_reload=False,
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    **kwargs
):
    cache_path = os.path.join(
        data_pth,
        f"preprocessed/{dataset_name}_data_{img_size}.pt"
    )

    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached {dataset_name}...")
        data = torch.load(cache_path)
    else:
        print(f"Processing {dataset_name} from source...")

        dataset_cls = get_dataset_class(dataset_name)

        dataset = dataset_cls(
            path_raw=os.path.join(data_pth, "raw", dataset_name),
            img_size=img_size,
            **kwargs
        )

        data_tuple = dataset.load()

        print(f"Ended processing {dataset_name} from source...")

        # Egységes dict formátum
        data = {
            "x": data_tuple[0],
            "y": data_tuple[1],
            "n_classes": data_tuple[2],
            "n_channels": data_tuple[3],
        }

        torch.save(data, cache_path)

    # ===== közös feldolgozás =====
    n_classes = data["n_classes"]
    n_channels = data["n_channels"]
    labels = data["y"]

    full_dataset = get_dataset_dataset(
        dataset_name,
        data
    )

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
        "class_num": n_classes,
        "labels": labels,
        "n_channels": n_channels
    }

    return train_dataset, val_dataset, test_dataset, meta


def get_dataset_dataset(name, data):
    if name == "ISIC2019":
        return dataLoading_ISIC2019(data)
    elif name == "MNIST":
        return dataLoading_MNIST(data)
    elif name == "AGNews":
        return dataLoading_AGNews(data)
    elif name == "DBpedia":
        return dataLoading_DBPedia(data)
    else:
        raise ValueError("Unknown dataset: {}".format(name))


def dataLoading_ISIC2019(data):
    images = data["x"]
    labels = data["y"]
    class_num = data["n_classes"]
    n_channels = data["n_channels"]

    return ImageDatasetInMemory(
        images=images,
        labels=labels,
        class_num=class_num,
        n_channels=n_channels
    )


def dataLoading_MNIST(data):
    images = data["x"]
    labels = data["y"]
    class_num = data["n_classes"]
    n_channels = data["n_channels"]

    return ImageDatasetInMemory(
        images=images,
        labels=labels,
        class_num=class_num,
        n_channels=n_channels
    )


def dataLoading_AGNews(data):
    x = data["x"]
    labels = data["y"]
    class_num = data["n_classes"]

    encodedtexts = x["input_ids"]
    attention_masks = x["attention_mask"]

    return TextDatasetInMemory(
        texts=encodedtexts,
        attention_masks=attention_masks,
        labels=labels,
        class_num=class_num
    )


def dataLoading_DBPedia(data):
    x = data["x"]
    labels = data["y"]
    class_num = data["n_classes"]

    encodedtexts = x["input_ids"]
    attention_masks = x["attention_mask"]

    return TextDatasetInMemory(
        texts=encodedtexts,
        attention_masks=attention_masks,
        labels=labels,
        class_num=class_num
    )