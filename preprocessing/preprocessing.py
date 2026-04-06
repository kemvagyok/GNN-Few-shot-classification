import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from .baseImageDataset import BaseImageDataset
from .baseTextDataset import BaseTextDataset

import argparse

def apply_remedial(df, label_cols):
    label_counts = df[label_cols].sum()
    irl = label_counts.max() / label_counts
    threshold = irl.mean()

    majority = irl[irl <= threshold].index
    minority = irl[irl > threshold].index

    mask = (df[majority].sum(axis=1) > 0) & (df[minority].sum(axis=1) > 0)

    safe = df[~mask]
    mixed = df[mask]

    if mixed.empty:
        return safe

    maj_only = mixed.copy()
    maj_only[minority] = 0

    min_only = mixed.copy()
    min_only[majority] = 0

    return pd.concat([safe, maj_only, min_only], ignore_index=True)

class ISIC2019Dataset(BaseImageDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = os.path.join(self.path, "ISIC_2019_Training_GroundTruth.csv")

    def load(self, train_size=4000, test_size=4000):
        df = pd.read_csv(self.csv_path)

        # class nevek (oszlopok a képen kívül)
        classes = list(df.columns[1:])
        print("Classes:", classes)

        paths, labels = [], []

        for _, row in df.iterrows():
            img_name = row["image"]

            # label = one-hot → index
            label = np.argmax(row[classes].values)

            # fájl elérési út
            img_path = os.path.join(self.img_dir, img_name + ".jpg")

            if os.path.exists(img_path):
                paths.append(img_path)
                labels.append(label)
            else:
                print(f"Missing: {img_path}")

        paths = np.array(paths)
        labels = np.array(labels)

        train_p, val_p, test_p, train_y, val_y, test_y = self.split_data(paths, labels)

        # limit size
        train_p, train_y = train_p[:train_size], train_y[:train_size]
        val_p, val_y = val_p[:int(train_size * 0.4)], val_y[:int(train_size * 0.4)]
        test_p, test_y = test_p[:test_size], test_y[:test_size]

        print("Loading ISIC train...")
        train_x, train_y = self.load_images(train_p, train_y)

        print("Loading ISIC val...")
        val_x, val_y = self.load_images(val_p, val_y)

        print("Loading ISIC test...")
        test_x, test_y = self.load_images(test_p, test_y)

        return train_x, train_y, val_x, val_y, test_x, test_y, len(classes), (1 if self.grayscale else 3)
    
class ChestXDataset(BaseImageDataset):
    def __init__(self,
                 multilabel=False,
                 **kwargs):
    
        super().__init__(**kwargs)
        self.csv_path = os.path.join(self.path, "Data_Entry_2017.csv")
        self.train_index_path = os.path.join(self.path, "train_val_list.txt")
        self.test_index_path = os.path.join(self.path, "test_list.txt")
        self.multilabel = multilabel

    def load(self, train_size=4000, test_size=4000):
        df = pd.read_csv(self.csv_path, usecols=["Image Index", "Finding Labels"])
        df.columns = ["img", "labels"]

        df["labels"] = df["labels"].str.split("|")

        classes = sorted({l for sub in df["labels"] for l in sub})
        label2idx = {l: i for i, l in enumerate(classes)}

        if self.multilabel:
            df["labels"] = df["labels"].apply(lambda x: [label2idx[i] for i in x])
            mlb = MultiLabelBinarizer()
            binarized = mlb.fit_transform(df["labels"])
            df = pd.concat([df["img"], pd.DataFrame(binarized)], axis=1)
            df = apply_remedial(df, df.columns[1:])
        else:
            df["labels"] = df["labels"].apply(lambda x: label2idx[x[0]])

        train_idx = pd.read_csv(self.train_index_path, header=None)[0]
        test_idx = pd.read_csv(self.test_index_path, header=None)[0]

        train_df = df[df["img"].isin(train_idx)]
        test_df = df[df["img"].isin(test_idx)]

        train_df, val_df = train_test_split(train_df, train_size=0.6)

        def build(df, size):
            paths = [os.path.join(self.image_dir, i) for i in df["img"][:size]]
            labels = df["labels"][:size].tolist()
            return self.load_images(paths, labels)

        train_x, train_y = build(train_df, train_size)
        val_x, val_y = build(val_df, int(train_size * 0.4))
        test_x, test_y = build(test_df, test_size)

        return train_x, train_y, val_x, val_y, test_x, test_y, len(classes), (1 if self.grayscale else 3)

def load_dataset(name, **kwargs):
    if name == "ISIC2019":
        return ISIC2019Dataset(**kwargs)
    elif name == "ChestX":
        return ChestXDataset(**kwargs)
    elif name == "AGNews":
        return BaseTextDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":

    parsers =argparse.ArgumentParser()
    parsers.add_argument("-dataset", type=str, default="ISIC2019", help="Dataset to preprocess (e.g., 'ChestX')")
    parsers.add_argument("--train_size", type=int, default=4000, help="Number of training files to process")
    parsers.add_argument("--test_size", type=int, default=4000, help="Number of testing files to process")
    parsers.add_argument("--img_size", type=int, default=128, help="Size of the images")
    args = parsers.parse_args()
    dataset_name = args.dataset
    train_files_size = args.train_size
    test_files_size = args.test_size
    img_size = args.img_size

    print(f"Starting data preprocessing for {dataset_name} dataset...")

    dataset = load_dataset(dataset_name, img_size=img_size, grayscale=True, path_raw=f"./data/raw/{dataset_name}")

    print(f"Ending data preprocessing for {dataset_name} dataset...")
    print(f"Saving preprocessed data for {dataset_name} dataset...")
    train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels = dataset.load(
        train_size=train_files_size,
        test_size=test_files_size,
    )
    pt_filename = f"./data/preprocessed/{dataset_name}_data_{train_files_size}_{img_size}.pt"

    torch.save({
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "n_classes": n_classes,
        "n_channels": n_channels }, 
        pt_filename
    )
    print(f"Ending preprocessed data for {dataset_name} dataset...")
    print(f"Data preprocessing completed for {dataset_name} dataset.")