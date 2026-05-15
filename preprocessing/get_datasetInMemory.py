import os
import torch
from .preprocessing import get_dataset_class
from .image.imageDatasetInMemory import ImageDatasetInMemory
from .text.textDatasetInMemory import TextDatasetInMemory
from .tabular.tabularDatasetInMemory import TabularDatasetInMemory
from .indexedDataset import IndexedDataset
from .tabular.UNSWPreprocessing import UNSWPreprocessor
from .tabular.lendingClubPreprocessing import LendingClubPreprocessor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import Tuple

import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


def lendingclub_split_indices(
    df,
    train_size: int,
    val_size: int, 
    test_size: int, 
    seed: int
):

    date_col="issue_d"

    # ---------------------------------------------
    # ratio check
    # ---------------------------------------------
    total = (
        train_size
        + val_size
        + test_size
    )

    assert abs(total - 1.0) < 1e-6, \
        "Ratios must sum to 1"

    # ---------------------------------------------
    # copy
    # ---------------------------------------------
    temp_df = df.copy()

    # ---------------------------------------------
    # datetime conversion
    # ---------------------------------------------
    temp_df[date_col] = pd.to_datetime(
        temp_df[date_col],
        format="%b-%Y",
        errors="coerce"
    )

    # ---------------------------------------------
    # stable deterministic shuffle
    # ---------------------------------------------
    rng = np.random.default_rng(seed)

    temp_df["_rand"] = rng.random(len(temp_df))

    # ---------------------------------------------
    # sort by time
    # secondary sort: random
    # ---------------------------------------------
    temp_df = temp_df.sort_values(
        by=[date_col, "_rand"]
    )
    temp_df["_iloc_idx"] = np.arange(len(temp_df))
    sorted_indices = temp_df["_iloc_idx"].values
    #sorted_indices = temp_df.index.values

    # ---------------------------------------------
    # split positions
    # ---------------------------------------------
    n = len(sorted_indices)

    train_end = int(
        n * train_size
    )

    val_end = int(
        n * (train_size + val_size)
    )

    # ---------------------------------------------
    # splits
    # ---------------------------------------------
    train_idx = sorted_indices[:train_end]

    val_idx = sorted_indices[
        train_end:val_end
    ]

    test_idx = sorted_indices[val_end:]

    return train_idx, val_idx, test_idx


def stratified_split(labels: np.ndarray[int], 
                     train_size: int,
                      val_size: int, 
                      test_size: int, 
                      seed: int
                     ) -> Tuple[np.ndarray[int], np.ndarray[int], np.ndarray[int]]:
    
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
    dataset_name: str,
    data_pth: str,
    seed: int,
    img_size: int = 28,
    force_reload: bool=False,
    train_size: int = 0.7,
    val_size: int = 0.1,
    test_size: int = 0.2,
    **kwargs
):
    cache_path = os.path.join(
        data_pth,
        f"preprocessed/{dataset_name}_data_{img_size}.pt"
    )

    if os.path.exists(cache_path) and not force_reload and (not dataset_name == "UNSW"):
        
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

        if dataset_name not in ["UNSW", "LENDING_CLUB"]:
            torch.save(data, cache_path)
        
    full_dataset = get_dataset_dataset(
        dataset_name,
        data
    )
    # ===== közös feldolgozás ===== 
    n_classes = data["n_classes"]
    n_channels = data["n_channels"]
    labels = data["y"]
    input_size = -1 # Csak a tabuláris dataseteknél lesz értelme, a többi esetben a modell fogja kezelni a bemenet méretét

    if dataset_name in ["LendingClub"]:
        #IDŐSORREND MIATT NEM LEHET STRATIFIED SPLITET HASZNÁLNI, HANEM TEMPORAL SPLITET KELL CSINÁLNI
        train_idx, val_idx, test_idx = lendingclub_split_indices(
            data["x"],
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed
        )
    else:
        train_idx, val_idx, test_idx = stratified_split(
            labels=labels,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed
        )
    
    if dataset_name == "UNSW":
        train_df = full_dataset.df.iloc[train_idx]
        prep = UNSWPreprocessor()
        prep.fit(train_df)
        full_df = pd.DataFrame(full_dataset.df)
        X_all, y_all = prep.transform(full_df)
        full_dataset.add_transformed(torch.tensor(X_all), torch.tensor(y_all), len(np.unique(y_all)))
        input_size = X_all.shape[1]
    elif dataset_name == "LendingClub":
        prep = LendingClubPreprocessor()

        clean_df = prep._clean_dataframe(
            full_dataset.df
        )
        clean_df = clean_df[:100000] #csak az első 100k sort használjuk, hogy gyorsabb legyen a feldolgozás 

        train_idx, val_idx, test_idx = lendingclub_split_indices(
                clean_df,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=seed
            )
        
        train_df = clean_df.iloc[train_idx]
        prep.fit(train_df)
        full_df = pd.DataFrame(full_dataset.df)
        X_all, y_all = prep.transform(
            full_df
            )       
        
        #-------------
        full_dataset.add_transformed(
            torch.tensor(X_all), 
            torch.tensor(y_all, dtype=torch.long), 
            len(np.unique(y_all)))
        
        input_size = X_all.shape[1]

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


def get_dataset_dataset(name, data):
    if name == "ISIC2019":
        return dataLoading_ISIC2019(data)
    elif name == "MNIST":
        return dataLoading_MNIST(data)
    elif name == "AGNews":
        return dataLoading_AGNews(data)
    elif name == "DBpedia":
        return dataLoading_DBPedia(data)
    elif name == "UNSW":
        return dataLoading_UNSW(data)
    elif name == "LendingClub":
        return dataLoading_LendingClub(data)
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

def dataLoading_UNSW(data):
    df = data["x"]
    return TabularDatasetInMemory(
        df=df
    )

def dataLoading_LendingClub(data):
    df = data["x"]
    return TabularDatasetInMemory(
        df=df
    )