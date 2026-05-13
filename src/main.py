from configs import Config

import argparse

from utils import (
    seed,
    is_main_process,
    save_results
)

from pipelineBuilder import PipelineBuilder

def main():

    parsers = argparse.ArgumentParser()
    parsers.add_argument("-config_fn", type=str, default="mnist", help="Choosing a config name")
    parsers.add_argument("--run_id", type=int, default=0, help="ID for the current run (used for logging)") #SBATCH esetében a run_id-vel beállítjuk a seed-et.
    args = parsers.parse_args()
    config = Config(f"./configs/files/{args.config_fn}.yaml")

    config.seed = config.seed + args.run_id

    pipeline = (
        PipelineBuilder(config)
        .build_device()
        .build_data()
        .build_loss()
        .build_metrics()
        .get_pipeline()
    )

    seed.set_seed(config.seed)

    results = pipeline.run(args.run_id)

    if is_main_process():
        save_results(results, config, args.run_id)

if __name__ == "__main__":
    main()