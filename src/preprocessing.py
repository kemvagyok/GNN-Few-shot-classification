import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms

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

def build_transform(img_size=28, grayscale=True):

    mean, std = ([0.5], [0.5]) if grayscale else (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    return transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

class BaseDataset:
    def __init__(self, img_size=28, grayscale=True):
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

class ISIC2019Dataset(BaseDataset):
    def __init__(self, root_dir="../data/ISIC2019", **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir

    def load(self, train_size=4000, test_size=4000):

        classes = sorted(d for d in os.listdir(self.root_dir)
                         if os.path.isdir(os.path.join(self.root_dir, d)))

        class_to_idx = {c: i for i, c in enumerate(classes)}

        paths, labels = [], []

        for c in classes:
            folder = os.path.join(self.root_dir, c)
            for f in os.listdir(folder):
                if f.endswith((".jpg", ".png")):
                    paths.append(os.path.join(folder, f))
                    labels.append(class_to_idx[c])

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
    
class ChestXDataset(BaseDataset):
    def __init__(self, 
                 image_dir = "../data/ChestX/images", 
                 csv_path = "../data/ChestX/Data_Entry_2017.csv", 
                 train_index_path = "../data/ChestX/train_val_list.txt", 
                 test_index_path = "../data/ChestX/test_list.txt", 
                 multilabel=False, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.train_index_path = train_index_path
        self.test_index_path = test_index_path
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
    else:
        raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":

    parsers =argparse.ArgumentParser()
    parsers.add_argument("-dataset", type=str, default="ChestX", help="Dataset to preprocess (e.g., 'ChestX')")
    parsers.add_argument("--train_size", type=int, default=4000, help="Number of training files to process")
    parsers.add_argument("--test_size", type=int, default=4000, help="Number of testing files to process")
    parsers.add_argument("--img_size", type=int, default=128, help="Size of the images")
    args = parsers.parse_args()
    dataset_name = args.dataset
    train_files_size = args.train_size
    test_files_size = args.test_size
    img_size = args.img_size

    print(f"Starting data preprocessing for {dataset_name} dataset...")

    dataset = load_dataset(dataset_name, img_size=img_size, grayscale=True)

    train_x, train_y, val_x, val_y, test_x, test_y, n_classes, n_channels = dataset.load(
        train_size=train_files_size,
        test_size=test_files_size,
    )
    pt_filename = f"../data/{dataset_name}/{dataset_name}_data_{train_files_size}_{img_size}.pt"

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

    print(f"Data preprocessing completed for {dataset_name} dataset.")