from preprocessing.get_dataset import get_dataset
from preprocessing.get_datasetInMemory import get_datasetInMemory

from preprocessing.image.imagesDataModule import build_transform
from models import initalizeModels
from configs import Config
from utils import setup_device
from utils.loss_factory import build_loss
from utils.dataset_utils import build_indices
from training.trainerInMemory import TrainerInMemory
from training.trainerInMemory_DDP import TrainerInMemory_DDP
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
    if config.train_mode == "embedding_only":
        
    if not is_ddp:
        return TrainerInMemory(
        embedder=embedder,
        gnn=gnn,
        criterion=criterion,
        config=config,
        device=device,
        metric_fn=metrics
    )
    else:
        return TrainerInMemory_DDP(
            embedder=embedder,
            gnn=gnn,
            criterion=criterion,
            config=config,
            local_rank=local_rank,
            metric_fn=metrics
        )


def measure_training_time(trainer, train_data, val_data, test_data, is_ddp, local_rank=0):
    """
    Kiszámolja a betanítási időt, figyelembe véve a GPU aszinkron működését és a DDP rankokat.
    """
    # 1. GPU szinkronizáció a pontos indításhoz
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()

    # 2. Betanítás futtatása
    best_val = trainer.train(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data
    )

    # 3. GPU szinkronizáció a pontos leállításhoz
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 4. Eredmény kiírása csak a fő folyamaton (Non-DDP esetén mindig, DDP esetén ha rank == 0)
    if not is_ddp or (is_ddp and local_rank == 0):
        mode_str = "DDP" if is_ddp else "Single GPU / CPU"
        print(f"\n{'='*40}")
        print(f"⏱️ [{mode_str}] Futási idő: {elapsed_time:.2f} másodperc ({elapsed_time/60:.2f} perc)")
        print(f"🏆 Legjobb validációs metrika: {best_val:.4f}")
        print(f"{'='*40}\n")

    return best_val, elapsed_time

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


for max_label in config.train_images_per_class:
        K_hops = [None] if not config.use_minibatch else K_hop_list
        K_hops = [None] if config.train_mode == "embedding_only" else K_hop_list
        if max_label == -1:
             train_idx = build_indices(labels[train_dataset.indices], max_per_class=len(train_dataset.indices))
        train_idx = build_indices(labels[train_dataset.indices], max_per_class=max_label)
        train_dataset = IndexedDataset(train_dataset, train_idx)
        for K_hop in K_hops:
            with wandb_run(config, is_ddp, run_id, K_hop, max_label, None, train_size, val_size, test_size):
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

# A korábban megírt mérőfüggvény hívása
best_val, exec_time = measure_training_time(
    trainer=trainer,
    train_data=train_dataset,
    val_data=val_dataset,
    test_data=test_dataset,
    is_ddp=is_ddp,
    local_rank=local_rank if is_ddp else 0
)
