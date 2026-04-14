import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import datasets

from .image.imagesDataModule import ImagesDataModule
from .text.textDataModule import TextDataModule 

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


# =========================================================
# ISIC2019
# =========================================================
class ISIC2019Dataset(ImagesDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = os.path.join(self.path, "ISIC_2019_Training_GroundTruth.csv")

    def load(self):
        df = pd.read_csv(self.csv_path)
        classes = list(df.columns[1:])

        paths, labels = [], []

        for _, row in df.iterrows():
            img_name = row["image"]
            label = np.argmax(row[classes].values)
            img_path = os.path.join(self.img_dir, img_name + ".jpg")

            if os.path.exists(img_path):
                paths.append(img_path)
                labels.append(label)

        paths = np.array(paths)
        labels = np.array(labels)
        """
        train_p, val_p, test_p, train_y, val_y, test_y = self.split_data(paths, labels)

        train_p, train_y = train_p[:train_file_size], train_y[:train_file_size]
        val_p, val_y = val_p[:int(train_file_size * 0.4)], val_y[:int(train_file_size * 0.4)]
        test_p, test_y = test_p[:test_file_size], test_y[:test_file_size]

        train_x, train_y = self.load_images(train_p, train_y)
        val_x, val_y = self.load_images(val_p, val_y)
        test_x, test_y = self.load_images(test_p, test_y)
        """
        x, y = self.load_images(paths, labels)

        return x, y, len(classes), (1 if self.grayscale else 3)


# =========================================================
# ChestX
# =========================================================
class ChestXDataset(ImagesDataModule):
    def __init__(self, multilabel=False, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = os.path.join(self.path, "Data_Entry_2017.csv")
        self.train_index_path = os.path.join(self.path, "train_val_list.txt")
        self.test_index_path = os.path.join(self.path, "test_list.txt")
        self.multilabel = multilabel

    def load(self):
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

        #train_df = df[df["img"].isin(train_idx)]
        #test_df = df[df["img"].isin(test_idx)]
        
        #train_df, val_df = train_df.sample(frac=0.6, random_state=42), train_df.sample(frac=0.4, random_state=42)
        '''
        def build(df, size):
            paths = [os.path.join(self.img_dir, i) for i in df["img"][:size]]
            labels = df.iloc[:size, 1:].values if self.multilabel else df["labels"][:size].tolist()
            return self.load_images(paths, labels)
        '''
        paths = [os.path.join(self.img_dir, i) for i in df["img"]]
        labels = df.iloc[:, 1:].values if self.multilabel else df["labels"].tolist()
        x, y = self.load_images(paths, labels)

        return x, y, len(classes), (1 if self.grayscale else 3)

class MNISTDataset(ImagesDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        transform = self.transform

        train_dataset = datasets.MNIST(
            root=self.path,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root=self.path,
            train=False,
            download=True,
            transform=transform
        )

        x = torch.cat([
            torch.stack([img for img, _ in train_dataset]),
            torch.stack([img for img, _ in test_dataset])
        ])

        y = torch.cat([
            torch.tensor([label for _, label in train_dataset]),
            torch.tensor([label for _, label in test_dataset])
        ])

        n_classes = 10
        n_channels = 1

        return x, y, n_classes, n_channels

class AGNewsDataset(TextDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = os.path.join(self.path, "train.csv")

    def load(self):
        df = pd.read_csv(self.csv_path, header=None,
                        names=["Class Index", "Title", "Description"],
                        skiprows=1)

        df['Text'] = df['Title'] + " " + df['Description']

        encode_texts, attention_masks = self.encode_texts(df['Text'].tolist())

        x = {
            "input_ids": encode_texts,
            "attention_mask": attention_masks
        }

        y = torch.tensor(df['Class Index'].values) - 1

        num_classes = len(y.unique())

        return x, y, num_classes, None
# =========================================================
# Factory
# =========================================================
def get_dataset_class(name):
    if name == "ISIC2019":
        return ISIC2019Dataset
    elif name == "ChestX":
        return ChestXDataset
    elif name == "MNIST":
        return MNISTDataset
    elif name == "AGNews":
        return AGNewsDataset
    else:
        raise ValueError("Unknown dataset: {}".format(name))