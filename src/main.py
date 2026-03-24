from train import run_training
from preproccesing.loadingModule import dataLoading_ChestX, dataLoading_ISIC2019, dataLoading_MNIST
from utils import setup_device, is_main_process, set_seed
from configs import Config
import argparse
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import wandb

def loading_dataset(dataset_name, **kwargs):
    if dataset_name == "MNIST":
        return dataLoading_MNIST(**kwargs)
    elif dataset_name == "ChestX":
        return dataLoading_ChestX(**kwargs)
    elif dataset_name == "ISIC2019":
        return dataLoading_ISIC2019(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument("-config_fn", type=str, default="isic2019", help="Choosing a config name")
    parsers.add_argument("--run_id", type=int, default=0, help="ID for the current run (used for logging)")
    args = parsers.parse_args()

    config_filename = args.config_fn
    run_id = args.run_id

    config = Config(f"./configs/{config_filename}.yaml")

    test_size = config.test_size
    dataset_path = config.dataset_path
    dataset_name = config.dataset_name
    results_path = config.results_path
    without_valid = config.without_valid
    K_hop_list = config.K_hop_list if  config.use_minibatch else [1]
    device, local_rank, is_ddp = setup_device()

    train_x, train_y, val_x, val_y, test_x, test_y, num_class, channel_size = \
        loading_dataset(config.dataset_name, data_pth=dataset_path)
    
    test_x_filtered = test_x[:test_size]
    test_y_filtered = test_y[:test_size]
    if without_valid:
        train_x = torch.cat([train_x, val_x], dim=0)
        train_y = torch.cat([train_y, val_y], dim=0)
        val_x = test_x[:test_size]
        val_y = test_y[:test_size]
        test_x_filtered = None 
        test_y_filtered = None

    best_results = []


    for K_hop in K_hop_list:
            for max_label in config.train_images_per_class:
                if is_main_process():
                    wandb.init(
                        project=f"few-shot-gnn-{dataset_name}_{'ddp' if is_ddp else 'without_ddp'}_{'minibatch' if config.use_minibatch else 'fullbatch'}",
                        name=f"run_{run_id}",
                        group=f"k_hop_{K_hop}",
                        config={
                            "dataset": dataset_name,
                            "K_hop": K_hop,
                            "max_label": max_label,
                            "epochs": config.epochs_max,
                            "lr_cnn": config.lr_cnn,
                            "lr_gcn": config.lr_gcn,
                            "batch_size": config.batch_size,
                            "K_neigh": config.K_neigh,
                            "is_ddp": is_ddp
                            },
                            reinit = True
                    )

                run_accs = []
                set_seed(42 + run_id)

                acc = run_training(
                    train_x, train_y,
                    val_x, val_y,
                    test_x_filtered, test_y_filtered,
                    num_class, channel_size,
                    K_hop, max_label,
                    config, device,
                    is_ddp = is_ddp
                )

                run_accs.append(acc)

                acc = np.mean(run_accs)
                best_results.append((K_hop, max_label, acc))


    if is_main_process():
        wandb.finish()


    if is_main_process():
        results_df = pd.DataFrame(best_results, columns=["K_hop", "max_label", "acc"])
        file_name = f"{results_path}/{dataset_name}_run{run_id}.csv"
        results_df.to_csv(file_name, index=False)
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()