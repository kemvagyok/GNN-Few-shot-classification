import os
import kaggle

os.environ["KAGGLE_USERNAME"] = "benedeksgi"
os.environ["KAGGLE_API_TOKEN"] = "KGAT_16acdd4df948ff735a12bf08b3884e2e"
print("Start downloading dataset from Kaggle...")
dataset_path = "amananandrai/ag-news-classification-dataset"
os.system(f"kaggle datasets download -d {dataset_path}")
#kagglehub.dataset_download("salviohexia/isic-2019-skin-lesion-images-for-classification", output_dir='./data/skinlesion')
