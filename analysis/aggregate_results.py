import pandas as pd
import glob
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("-dataset", type=str, default="mnist", help="Dataset name (e.g., 'MNIST')")
args.add_argument("-metric", type=str, default="accuracy", help="Metric name")
args.add_argument("-train_type", type=str, default="embedding&GNN", help="Train type name")

dataset_name = args.parse_args().dataset
metric = args.parse_args().metric
train_type = args.parse_args().train_type
print(f"train_type: {train_type}")
print(f"Dataset name: {dataset_name}")
files = glob.glob(f"./results/{dataset_name}/{dataset_name}_{metric}__{train_type}_run*.csv")
print(files)
dfs = [pd.read_csv(f) for f in files]
all_results = pd.concat(dfs)

final = all_results.groupby(
    ["K_hop", "max_label"]
)["acc"].agg(["mean","std"]).reset_index()

final.to_csv(f"./results/{dataset_name}/final_results_{dataset_name}_{train_type}.csv", index=False)

print(final)