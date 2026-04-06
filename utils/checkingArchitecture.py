# =========================
# Standard library
# =========================
import os
import random
import argparse

# =========================
# Third-party libraries
# =========================
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import wandb
from networkx import subgraph

# =========================
# PyTorch core
# =========================
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# =========================
# PyTorch Geometric
# =========================
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# =========================
# Saját modulok
# =========================
from configs import Config

from graph_tools import create_edge_index
from models import cnnModel, gcnModel, resnetModel
from preproccesing.loadingModule  import (
    dataLoading_MNIST,
    dataLoading_ChestX,
    dataLoading_ISIC2019,
    traindatasetFiltering
)
from preproccesing.preprocessing import ISIC2019Dataset, load_dataset

def checkingDataPathAvaible(data_pth, dataset_name, file_size, img_size):
    assert os.path.exists(data_pth), f"Data path {data_pth} does not exist. Please check the path and try again."
    #RAW
    assert os.path.exists(os.path.join(data_pth, "raw", dataset_name)), f"{dataset_name}/raw/{dataset_name} dataset folder not found in {data_pth}. Please ensure the ISIC2019 dataset is placed in the correct directory."
    #PREPROCCESSED
    assert os.path.isfile(os.path.join(data_pth, "preprocessed" ,f"{dataset_name}_data_{file_size}_{img_size}.pt")), f"Preprocessed data file ISIC2019_data_{file_size}_{img_size}.pt not found in {data_pth}/ISIC2019. Please run the preprocessing script to generate the data file."

def checkingEmbeddingModel(data_pth):
    config = Config("./configs/mnist_without_dpp.yaml")
    train_x, train_y, val_x, val_y,test_x, test_y, n_classes, n_channels = dataLoading_ISIC2019(
        data_pth=data_pth,
        files_size=4000,
        img_size=128,
        force_reload=False,
        grayscale=True
    )
    print("Train images shape:", train_x.shape)
    print("Train labels shape:", train_y.shape)
    print("Validation images shape:", val_x.shape)
    print("Validation labels shape:", val_y.shape)

    train_x = torch.cat([train_x, val_x], dim=0)
    train_y = torch.cat([train_y, val_y], dim=0)

    print("Train images shape:", train_x.shape)
    print("Train labels shape:", train_y.shape)
    model_cnn = cnnModel(channel_size=n_channels)
    model_resnet = resnetModel(output_dim=64, in_channels=n_channels)
    model_cnn.eval()
    model_resnet.eval()

    with torch.no_grad():
        sample_imgs = train_x[:16]
        sample_labels = train_y[:16]
        embeddings_cnn = model_cnn(sample_imgs)
        embeddings_resnet = model_resnet(sample_imgs)

    assert embeddings_cnn.shape == (16, config.latens_size), f"Expected CNN embeddings shape (16, {config.latens_size}), got {embeddings_cnn.shape}"
    assert embeddings_resnet.shape == (16, config.latens_size), f"Expected ResNet embeddings shape (16, {config.latens_size}), got {embeddings_resnet.shape}"
    
    print("Sample labels:", sample_labels)
    print("Embeddings shape:", embeddings_cnn.shape)
    print("Embeddings:", embeddings_cnn)
    print("Embeddings shape:", embeddings_resnet.shape)
    print("Embeddings:", embeddings_resnet)

def checkingGCNModel():
    num_nodes = 10
    num_features = 16
    num_classes = 3

    x = torch.randn((num_nodes, num_features))
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)

    model_gcn = gcnModel(num_features=num_features, num_classes=num_classes)
    model_gcn.eval()

    with torch.no_grad():
        output = model_gcn(data)

    print("GCN output shape:", output.shape)
    print("GCN output:", output)




if __name__ == "__main__":
    config = Config("./configs/isic2019.yaml")
    dataset_name = config.dataset_name
    data_pth = config.dataset_path
    results_pth = config.results_path
    logs_pth = config.logs_path
    checkingDataPathAvaible(data_pth = data_pth, dataset_name= dataset_name, file_size=4000, img_size=128)
    checkingEmbeddingModel(data_pth = data_pth)
    checkingGCNModel()