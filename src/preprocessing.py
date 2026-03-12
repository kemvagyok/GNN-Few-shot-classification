import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms


def _build_transform(image_size: int, grayscale: bool):
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),          # [0,255] → [0.0,1.0], [C,H,W]
    ]
    if grayscale:
        # ToTensor() után 1 csatorna van; normalizálás 1 csatornás statisztikával
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)

def fileLoading_ChestX(
        image_dir="data/nih_chest_xray/images",
        csv_path="data/nih_chest_xray/Data_Entry_2017.csv",
        train_index_path = "data/nih_chest_xray/train_val_list.txt",
        test_index_path = "data/nih_chest_xray/test_list.txt",
        img_size = 28,
        max_samples_per_class = None,
        train_files_size = 100,
        test_files_size = 100,
        isMultilabel = False,
        grayscale = True):

    df = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])
    df = df.rename(columns = 
        {"Image Index" : "img_idx",
         "Finding Labels" : "labels"})
    df["labels"] =  df["labels"].str.split("|")
    #df["labels_list"] = df["Finding Labels"].str.split("|").apply(lambda tags: [t.strip() for t in tags])
    classes = df["labels"].apply(lambda tags: [t.strip() for t in tags]).explode().unique()
    
    label2idx = {label: idx for idx, label in enumerate(classes)}

    if isMultilabel:
        df["labels"] = df["labels"].apply(lambda labels: [label2idx[label] for label in labels])
        mlb = MultiLabelBinarizer()
        df["labels"] = mlb.fit_transform(df["labels"])
    else:
        df["labels"] = df["labels"][:].apply(lambda labels: label2idx.get(labels[0]))
    


    train_index = pd.read_csv(train_index_path, header = None)
    test_index = pd.read_csv(test_index_path, header = None)



    train_df = df[df["img_idx"].isin(train_index.iloc[:, 0])]
    test_df = df[df["img_idx"].isin(test_index.iloc[:, 0])]

    train_tensors = []
    train_targets = []
    test_tensors = []
    test_targets = []
    tf = _build_transform(img_size, grayscale)
    def images_load(df, size):
        tensors = [] 
        targets = []
        for row in df[:size].itertuples():
            img_path = os.path.join(image_dir, row.img_idx)
            img = Image.open(img_path).convert("L" if grayscale else "RGB")
            img = tf(img)
            tensors.append(img)
            targets.append(row.labels)
        return tensors, targets
    

    train_tensors, train_targets = images_load(train_df, train_files_size)
    test_tensors, test_targets = images_load(test_df, test_files_size)

    train_x = torch.stack(train_tensors, dim = 0)
    test_x = torch.stack(test_tensors, dim = 0)
    if not grayscale:
        train_x = train_x.permute(0,3,1,2)
        test_x = test_x.permute(0,3,1,2)
        
    train_y = torch.tensor(train_targets)
    test_y = torch.tensor(test_targets)

    n_classes = len(classes)
    n_channels = 1 if grayscale else 3
    
    return (train_x, train_y, test_x, test_y, n_classes, n_channels)


if __name__ == "__main__":
    print("Starting data preprocessing for ChestX dataset...")
    train_x, train_y, test_x, test_y, n_classes, n_channels = fileLoading_ChestX()
    torch.save({
    "train_x": train_x,
    "train_y": train_y,
    "test_x": test_x,
    "test_y": test_y,
    "n_classes": n_classes,
    "n_channels": n_channels
}, "data/nih_chest_xray/ChestX_data.pt")
    print("Data preprocessing completed for ChestX dataset.")