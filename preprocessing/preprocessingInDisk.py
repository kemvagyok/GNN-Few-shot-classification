import os
import numpy as np
import pandas as pd

class ISIC2019Preprocessing():
    def load(self, csv_path, img_dir):
        df = pd.read_csv(csv_path)
        classes = list(df.columns[1:])
        images_name, labels, img_paths = [], [], []
        for _, row in df.iterrows():
            img_name = row["image"] + ".jpg"
            label = np.argmax(row[classes].values)
            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(img_path):
                images_name.append(img_name)
                labels.append(label)
                img_paths.append(img_path)

        images_name = np.array(images_name)
        labels = np.array(labels)
        n_channels = 3

        return images_name, labels, len(classes), n_channels
