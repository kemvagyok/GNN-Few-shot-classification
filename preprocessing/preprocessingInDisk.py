import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import datasets, transforms
from transformers import AutoTokenizer
import torchvision.transforms.functional as F
from .image.imagesDataModule import ImagesDataModule
from .text.textDataModule import TextDataModule 
import tqdm
from PIL import Image
 
def load_images(paths, labels, grayscale, transform=None):
    imgs = []
    if transform is None:
        transform = F
    for p in tqdm.tqdm(paths):
        img = Image.open(p).convert("L" if grayscale else "RGB")
        imgs.append(transform(img))
    return imgs


class ISIC2019Preprocessing():
    def load(self, csv_path, img_dir):
        df = pd.read_csv(csv_path)
        classes = list(df.columns[1:])
        images_name, labels, img_paths = [], [], []
        saving = True
        for _, row in df.iterrows():
            img_name = row["image"] + ".jpg"
            label = np.argmax(row[classes].values)
            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(img_path):
                images_name.append(img_name)
                labels.append(label)
                img_paths.append(img_path)

        images_name = np.array(images_name)
        labels = np.array(labels)
        n_channels = 3
        if saving:
            imgs = load_images(img_paths)
            torch.save({
            "imgs": imgs,
            "labels": labels,
            }, f"{img_dir}_{datasets}.pt")
            torch.stack(imgs), torch.tensor(labels)
        return images_name, labels, len(classes), n_channels

class MNISTPreprocessing():
    def load(self, path, transform=None):
        train_dataset = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transform
        )

        images = torch.cat([
            torch.stack([img for img, _ in train_dataset]),
            torch.stack([img for img, _ in test_dataset])
        ])

        labels = torch.cat([
            torch.tensor([label for _, label in train_dataset]),
            torch.tensor([label for _, label in test_dataset])
        ])

        n_classes = 10
        n_channels = 1

        return images, labels, n_classes, n_channels

class AGNewsPreprocessing():
    def load(self, csv_train_path, csv_test_path):
        df_train = pd.read_csv(csv_train_path, header=None,
                              names=["Class Index", "Title", "Description"],
                              skiprows=1)
        df_test = pd.read_csv(csv_test_path, header=None,
                             names=["Class Index", "Title", "Description"],
                             skiprows=1)
        df_train['Text'] = df_train['Title'] + " " + df_train['Description']
        df_test['Text'] = df_test['Title'] + " " + df_test['Description']

        encode_texts, attention_masks = encode_texts(
            texts=df_train['Text'].tolist() + df_test['Text'].tolist(),
            max_len=128,
            tokenizer_name="bert-base-uncased"
        )

        y_train = torch.tensor(df_train['Class Index'].values) - 1
        y_test = torch.tensor(df_test['Class Index'].values) - 1
        labels = torch.cat([y_train, y_test])
        num_classes = len(labels.unique())

        return encode_texts, attention_masks, labels, num_classes, None

class DBpediaPreprocessing():
    def load(self, csv_train_path, csv_test_path):
        # Beolvasás a fejlécnevek alapján: 'label', 'title', 'content'
        df_train = pd.read_csv(csv_train_path)
        df_test = pd.read_csv(csv_test_path)
        # Szövegek összefűzése: a 'title' és a 'content' oszlopokat használjuk
        # (Az eredeti kódban 'Description' volt, itt 'content')
        df_train['Text'] = df_train['title'].fillna('') + " " + df_train['content'].fillna('')
        df_test['Text'] = df_test['title'].fillna('') + " " + df_test['content'].fillna('')

        # Tokenizálás/Enkódolás (a szülőosztály self.encode_texts metódusát használva)
        encode_texts, attention_masks = encode_texts(
            texts=df_train['Text'].tolist() + df_test['Text'].tolist(),
            max_len=128,
            tokenizer_name="bert-base-uncased"
        )

        combined_labels = pd.concat([df_train['label'], df_test['label']]).values
        labels = torch.tensor(combined_labels)
        num_classes = len(labels.unique())

        # Visszatérünk az adatokkal (x, y, osztályok száma, validációs adatok)
        return encode_texts, attention_masks, labels, num_classes, None

def encode_texts(self,  texts, max_len=128, tokenizer_name="bert-base-uncased",):
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return encodings['input_ids'], encodings['attention_mask']
