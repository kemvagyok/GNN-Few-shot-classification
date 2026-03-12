import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

def ChestXLoading(
        image_dir="data/nih_chest_xray/images"
        csv_path="data/nih_chest_xray/Data_Entry_2017.csv"
        train_index_path = "data/nih_chest_xray/train_val_list.txt"
        test_index_path = "data/nih_chest_xray/test_list.txt",
        img_size=(28,28),
        max_samples_per_class=None,
        train_files_size = 4000,
        test_files_size = 4000,
        isMultilabel = False):

    df = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])
    df = df.rename(columns = 
        {"Image Index" : "img_idx",
         "Finding Labels" : "labels"})
    df["labels"] =  df["labels"].str.split("|")
    #df["labels_list"] = df["Finding Labels"].str.split("|").apply(lambda tags: [t.strip() for t in tags])
    classes = df["labels"].apply(lambda tags: [t.strip() for t in tags]).explode().unique()
    #classes = df["labels_list"].explode().unique()
    
    label2idx = {label: idx for idx, label in enumerate(classes)}

    if isMultilabel:
        df["labels"] = df["labels"].apply(lambda labels: [label2idx[label] for label in labels])
        mlb = MultiLabelBinarizer()
        df["labels"] = mlb.fit_transform(df["labels_idx"])
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

    def images_load(df, size):
        tensors = [] 
        targets = 
        for row in df[:size].itertuples():
            img_path = os.path.join(image_dir, row.img_idx)
            img = Image.open(img_path).convert("L" if grayscale else "RGB")
            tensors.append(torch.tensor(np.asarray(img)))
            targets.append(row.labels)
        return tensors, targets
        
    train_tensors, train_targets = images_load(train_df, train_files_size)
    test_tensors, test_targets = images_load(test_df, test_files_size)

    X_train = torch.stack(train_tensors, dim = 0).unsqueeze(1)
    X_test = torch.stack(test_tensors, dim = 0).unsqueeze(1)

    if grayscale:
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)
    else:
        X_train = X_train.permute(0,3,1,2)
        X_test = X_test.permute(0,3,1,2)
        
    Y_train = torch.tensor(train_targets)
    Y_test = torch.tensor(test_targets)

    return (X_train, Y_train), (X_test, Y_test), label2idx