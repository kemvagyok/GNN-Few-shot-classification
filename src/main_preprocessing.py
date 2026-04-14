from preprocessing.loadingModule import load_dataset_cached
# --- Project modules ---
from configs import Config
import argparse

config_filename = "isic2019"
config = Config(f"./configs/{config_filename}.yaml")

train_x, train_y, val_x, val_y, test_x, test_y, num_class, channel_size = \
    load_dataset_cached(
        dataset_name=config.dataset_name,
        data_pth=config.dataset_path,
        img_size=28,
        files_size=-1
    )

print(len(train_x), len(train_y), len(val_x), len(val_y), len(test_x), len(test_y), num_class, channel_size)