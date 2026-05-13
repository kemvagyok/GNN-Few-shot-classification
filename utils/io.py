import os
import pandas as pd

def save_results(results, config, run_id):
    result_dataset_path = os.path.join(config.results_path, config.dataset_name)
    print(f"Saving results to {result_dataset_path}...")
    if not os.path.isdir(result_dataset_path):
        os.mkdir(result_dataset_path)
    results_df = pd.DataFrame(results, columns=["K_hop", "max_label", "acc"])
    if config.train_mode == "embedding_only":
        file_name = f"{result_dataset_path}/{config.dataset_name}_{config.metrics}_onlyEmbedding_run{run_id}.csv"
    else:
        file_name = f"{result_dataset_path}/{config.dataset_name}_{config.metrics}__embedding&GNN_run{run_id}.csv"
    results_df.to_csv(file_name, index=False)