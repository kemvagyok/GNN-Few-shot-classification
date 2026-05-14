from preprocessing.loadingModule import load_dataset_cached
# --- Project modules ---
from configs import Config
import argparse

config_filename = "isic2019"
config = Config(f"./configs/{config_filename}.yaml")

