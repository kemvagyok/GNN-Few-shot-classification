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
class ISIC2019_loading(ImagesDataModule):
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

        x, y = self.load_images(paths, labels)

        return x, y, len(classes), (1 if self.grayscale else 3)


# =========================================================
# ChestX
# =========================================================
class ChestX_loading(ImagesDataModule):
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

        paths = [os.path.join(self.img_dir, i) for i in df["img"]]
        labels = df.iloc[:, 1:].values if self.multilabel else df["labels"].tolist()
        x, y = self.load_images(paths, labels)

        return x, y, len(classes), (1 if self.grayscale else 3)
# =========================================================
# MNIST
# =========================================================
class MNIST_loading(ImagesDataModule):
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
# =========================================================
# AGNews
# =========================================================
class AGNews_loading(TextDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_train_path = os.path.join(self.path, "train.csv")
        self.csv_test_path = os.path.join(self.path, "test.csv")

    def load(self):
        df_train = pd.read_csv(self.csv_train_path, header=None,
                              names=["Class Index", "Title", "Description"],
                              skiprows=1)
        df_test = pd.read_csv(self.csv_test_path, header=None,
                             names=["Class Index", "Title", "Description"],
                             skiprows=1)
        df_train['Text'] = df_train['Title'] + " " + df_train['Description']
        df_test['Text'] = df_test['Title'] + " " + df_test['Description']

        encode_texts, attention_masks = self.encode_texts(df_train['Text'].tolist() + df_test['Text'].tolist())

        x = {
            "input_ids": encode_texts,
            "attention_mask": attention_masks
        }

        y_train = torch.tensor(df_train['Class Index'].values) - 1
        y_test = torch.tensor(df_test['Class Index'].values) - 1
        y = torch.cat([y_train, y_test])
        num_classes = len(y.unique())

        return x, y, num_classes, None
# =========================================================
# DBpedia
# =========================================================
class DBpedia_loading(TextDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Beállítjuk a fájl elérési útját (a te esetedben test.csv)
        self.csv_train_path = os.path.join(self.path, "train.csv")
        self.csv_test_path = os.path.join(self.path, "test.csv")

    def load(self):
        # Beolvasás a fejlécnevek alapján: 'label', 'title', 'content'
        df_train = pd.read_csv(self.csv_train_path)
        df_test = pd.read_csv(self.csv_test_path)
        # Szövegek összefűzése: a 'title' és a 'content' oszlopokat használjuk
        # (Az eredeti kódban 'Description' volt, itt 'content')
        df_train['Text'] = df_train['title'].fillna('') + " " + df_train['content'].fillna('')
        df_test['Text'] = df_test['title'].fillna('') + " " + df_test['content'].fillna('')

        # Tokenizálás/Enkódolás (a szülőosztály self.encode_texts metódusát használva)
        encode_texts, attention_masks = self.encode_texts(df_train['Text'].tolist() + df_test['Text'].tolist())

        x = {
            "input_ids": encode_texts,
            "attention_mask": attention_masks
        }

        
        combined_labels = pd.concat([df_train['label'], df_test['label']]).values
        y = torch.tensor(combined_labels)
        num_classes = len(y.unique())

        # Visszatérünk az adatokkal (x, y, osztályok száma, validációs adatok)
        return x, y, num_classes, None

# =========================================================
# UNSW
# =========================================================

class UNSW_loading:
    def __init__(self, path_raw, **kwargs):
        self.path = path_raw
        self.csv_train_path = os.path.join(self.path, "UNSW_NB15_training-set.csv")
        self.csv_test_path = os.path.join(self.path, "UNSW_NB15_testing-set.csv")
    def load(self):
        train_df = pd.read_csv(self.csv_train_path)
        test_df  = pd.read_csv(self.csv_test_path)
        df = pd.concat((train_df,test_df))
        labels = df['label'].values
        num_classes = len(df['label'].value_counts())
        return df, labels, num_classes, None
# =========================================================
# 
# =========================================================

class LendingClub_loading:
    def __init__(self, path_raw, **kwargs):
        self.path = path_raw
        #self.csv_accepted_path = os.path.join(self.path, "accepted_2007_to_2018Q4.csv")
        self.csv_accepted_path = os.path.join(self.path, "proba.csv")
        #self.csv_rejected_path = os.path.join(self.path, "rejected_2007_to_2018Q4.csv")
    
    def load(self):
        train_df = pd.read_csv(self.csv_accepted_path, low_memory=False)
        #test_df  = pd.read_csv(self.csv_rejected_path)
        #df = pd.concat((train_df,test_df))
        df = train_df

        labels = None

        if "loan_status" in df.columns:
            labels = (
                df["loan_status"] == "Charged Off"
            ).astype(np.float32).values


        num_classes = len(np.unique_counts(labels))


        return df, labels, num_classes, None

# =========================================================
# Factory
# =========================================================
def get_dataset_class(name):
    if name == "ISIC2019":
        return ISIC2019_loading
    elif name == "ChestX":
        return ChestX_loading
    elif name == "MNIST":
        return MNIST_loading
    elif name == "AGNews":
        return AGNews_loading
    elif name == "DBpedia":
        return DBpedia_loading
    elif name == "UNSW":
        return UNSW_loading
    elif name == "LendingClub":
        return LendingClub_loading
    else:
        raise ValueError("Unknown dataset: {}".format(name))