import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "./data/raw/ISIC2019"  # igazítsd!
CSV_PATH = os.path.join(DATA_PATH, "ISIC_2019_Training_GroundTruth.csv")
IMG_DIR = os.path.join(DATA_PATH, "images")  # ha így hívod

def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


def plot_class_distribution(df):
    classes = df.columns[1:]
    counts = df[classes].sum()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45)
    plt.title("Class Distribution (ISIC2019)")
    plt.ylabel("Number of samples")
    plt.show()

    print(counts.sort_values(ascending=False))

def check_multilabel(df):
    classes = df.columns[1:]
    label_counts_per_row = df[classes].sum(axis=1)

    print("Rows with multiple labels:", (label_counts_per_row > 1).sum())
    print("Rows with zero labels:", (label_counts_per_row == 0).sum())

def check_missing_images(df):
    missing = 0

    for _, row in df.iterrows():
        img_name = row["image"]
        path = os.path.join(IMG_DIR, img_name + ".jpg")

        if not os.path.exists(path):
            missing += 1

    print(f"Missing images: {missing}")

from PIL import Image
import numpy as np

def image_size_stats(df, sample_size=500):
    sizes = []

    sample_df = df.sample(sample_size)

    for _, row in sample_df.iterrows():
        path = os.path.join(IMG_DIR, row["image"] + ".jpg")

        if os.path.exists(path):
            img = Image.open(path)
            sizes.append(img.size)

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    print("Width mean:", np.mean(widths))
    print("Height mean:", np.mean(heights))

if __name__ == "__main__":
    df = load_data()

    plot_class_distribution(df)
    check_multilabel(df)
    check_missing_images(df)
    image_size_stats(df)
