from preprocessing.get_dataset import get_dataset
from preprocessing.get_datasetInMemory import get_datasetInMemory

from preprocessing.image.imagesDataModule import build_transform
from models import initalizeModels
from configs import Config
from utils import setup_device
from utils.loss_factory import build_loss
from utils.dataset_utils import build_indices
#----------------------------------------------------
from training.trainer import Trainer
from training.trainerDDP import TrainerDDP

from training.trainerEmbeddingGNNMinibatch import TrainerEmbeddingGNNMinibatch
from training.trainerEmbeddingGNNMinibatch_DDP import TrainerEmbeddingGNNMinibatch_DDP

from training.trainerEmbeddingOnlyMinibatch import TrainerEmbeddingOnlyMinibatch
from training.trainerEmbeddingOnlyMinibatch_DDP import TrainerEmbeddingOnlyMinibatch_DDP
#----------------------------------------------------
from preprocessing.indexedDataset import IndexedDataset
import numpy as np
import time
import torch
import torch.distributed as dist
import argparse
from utils.metric_factory import build_metrics
from utils import (
    setup_device,
    is_main_process,
    set_seed,
    graph_builder,
    save_results,
    get_class_distribution,
    print_distribution,
    sample_k_per_class,
    split_dataset,
    wandb_run
)

def get_trainer(is_ddp, embedder, gnn, criterion, metrics, config, device, local_rank):
    if config.embedding_minibatch:
        if not is_ddp:
            return TrainerEmbeddingGNNMinibatch(
            embedder=embedder,
            gnn=gnn,
            criterion=criterion,
            config=config,
            device=device,
            metric_fn=metrics
        )
        else:
            return TrainerEmbeddingGNNMinibatch_DDP(
                embedder=embedder,
                gnn=gnn,
                criterion=criterion,
                config=config,
                local_rank=local_rank,
                metric_fn=metrics,
                device=device
            )
    else:
        if not is_ddp:
            return Trainer(
            embedder=embedder,
            gnn=gnn,
            graph_builder=graph_builder,
            criterion=criterion,
            config=config,
            device=device,
            metric_fn=metrics
        )
        else:
            return TrainerDDP(
                embedder=embedder,
                gnn=gnn,
                graph_builder=graph_builder,
                criterion=criterion,
                config=config,
                local_rank=local_rank,
                metric_fn=metrics,
                device=device
            )


# ---------------- CONFIG ----------------

parsers = argparse.ArgumentParser()
parsers.add_argument("-config_fn", type=str, default="mnist", help="Choosing a config name")
parsers.add_argument("--run_id", type=int, default=0, help="ID for the current run (used for logging)")
args = parsers.parse_args()
config_filename = args.config_fn
run_id = args.run_id

# ---------------- CONFIG ----------------

config = Config(f"./configs/{config_filename}.yaml")
K_hop_list = config.K_hop_list
img_size = config.img_size
grayscale = True


train_size=0.7
val_size=0.1
test_size=0.2
# ---------------- DATA ----------------
"""
train_dataset, val_dataset, test_dataset, meta = get_dataset(
    data_pth=config.dataset_path,
    dataset_name=config.dataset_name,
    img_dir="images",
    transform=build_transform(
        grayscale=grayscale
    ),
    train_size=train_size,
    val_size=val_size,
    test_size=test_size
)
"""
train_dataset, val_dataset, test_dataset, meta = get_datasetInMemory(
            dataset_name=config.dataset_name,
            data_pth=config.dataset_path,
            img_size=28,
            files_size=config.files_size if hasattr(config, "files_size") else 4000,
            force_reload=False,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
#META:
# - n_classes
# - n_Channels
# - labels

# ---------------- SPLITTING ----------------
# labels a full datasetből
labels = meta["labels"]

# ---------------- DEVICE ----------------

device, local_rank, is_ddp = setup_device()

# ---------------- LOSS ----------------

criterion = build_loss(
    config=config,
    targets=meta["labels"],
    num_classes=meta["class_num"],
    device=device,
    weights=None
)

# -----------------------------------------

metrics = build_metrics(config)

# ---------------- TRAINER ----------------
best_results = []
train_dataset_original_indices = train_dataset.indices
for max_label in config.train_images_per_class:
        K_hops = [None] if (config.train_mode == "embedding_only" or (not config.gnn_minibatch)) else K_hop_list
        if max_label == -1:
             train_idx = build_indices(labels[train_dataset_original_indices], max_per_class=len(train_dataset_original_indices))
        train_idx = build_indices(labels[train_dataset_original_indices], max_per_class=max_label)
        train_dataset = IndexedDataset(train_dataset.base, train_idx)
        for K_hop in K_hops:
            with wandb_run(config, is_ddp, run_id, K_hop, max_label, train_size, val_size, test_size):
                # ---------------- MODELS ----------------
                embedder, gnn = initalizeModels(
                    config=config,
                    channel_size= 1 if grayscale else meta["n_channels"],
                    num_class=meta["class_num"],
                    latens_size=config.latens_size,
                    device=device,
                    is_ddp=False
                )
                # ---------------- TRAINER ----------------
                trainer = get_trainer(is_ddp, embedder, gnn, criterion, metrics, config, device, local_rank)

                metric_number = trainer.train(
                    train_dataset=train_dataset, 
                    val_dataset=(val_dataset if val_dataset is not None else None), 
                    test_dataset=(test_dataset if test_dataset is not None else None),
                    K_hop = K_hop
                    )

            best_results.append((K_hop, max_label, metric_number))

if is_main_process():
    save_results(best_results, config, run_id)

if is_ddp:
    dist.destroy_process_group()
