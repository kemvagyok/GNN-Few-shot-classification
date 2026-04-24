from preprocessing.loadingModule import load_dataset_cached
# --- Project modules ---
from configs import Config
import argparse

config_filename = "isic2019"
config = Config(f"./configs/{config_filename}.yaml")


load_dataset_cached(
    dataset_name=config.dataset_name,
    data_pth=config.dataset_path,
    img_size=224,
    files_size=-1
)

