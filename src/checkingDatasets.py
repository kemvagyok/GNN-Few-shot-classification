from preprocessing.loadingModule import load_dataset_cached
from utils import  get_class_distribution, print_distribution, wandb_run

train_x, train_y, val_x, val_y, test_x, test_y, num_class, channel_size = \
    load_dataset_cached(
        dataset_name="ISIC2019",
        data_pth="./data",
        img_size=128,
        files_size = 4000
    )



print_distribution(get_class_distribution(targets = train_y, num_classes = 8))
print_distribution(get_class_distribution(targets = val_y, num_classes = 8))
print_distribution(get_class_distribution(targets = test_y, num_classes = 8))
