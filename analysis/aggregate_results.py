import pandas as pd
import glob
import argparse
import os

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="mnist", help="Dataset name (e.g., 'ChestX')")

dataset_name = args.parse_args().dataset

files = glob.glob(f"../results/{dataset_name}/results_{dataset_name}_run*.csv")

dfs = [pd.read_csv(f) for f in files]

all_results = pd.concat(dfs)

final = all_results.groupby(
    ["K_hop","max_label"]
)["acc"].agg(["mean","std"]).reset_index()

final.to_csv(f"../results/dataset_name/final_results__{dataset_name}.csv", index=False)

print(final)