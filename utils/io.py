import os
import pandas as pd

def save_results(results, config, run_id):
    result_dataset_path = os.path.join(config.results_path, config.dataset_name)
    if not os.path.isdir(result_dataset_path):
        os.mkdir(result_dataset_path)
        results_df = pd.DataFrame(results, columns=["K_hop", "max_label", "acc"])
        file_name = f"{result_dataset_path}/{config.dataset_name}_run{run_id}.csv"
        results_df.to_csv(file_name, index=False)