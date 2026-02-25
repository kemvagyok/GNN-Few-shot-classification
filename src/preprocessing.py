import os
from PIL import Image
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, INFO, Evaluator



def dataLoading_MNIST():
  transform = ToTensor()
  train_dataset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)
  test_dataset = datasets.MNIST('./data',
                                train=False,
                                download=True,
                                transform=transform)

  n_classes = len(train_dataset.classes)
  n_channels = 1

  #--- Train
  train_x = train_dataset.data.unsqueeze(1) / 255
  train_y = train_dataset.targets
  #--- Test
  test_x = test_dataset.data.unsqueeze(1) / 255
  test_y = test_dataset.targets
  return (train_x, train_y, test_x, test_y, n_classes, n_channels)


def dataLoading_medMNIST():
  data_flag = 'dermamnist'
  download = True

  info = INFO[data_flag]

  n_classes = len(info['label'])
  n_channels = info['n_channels']


  transform = ToTensor()

  train_dataset = DermaMNIST(split="train",
                             download=True,
                             size = 28,
                             transform=transform)
  test_dataset = DermaMNIST(split="test",
                            download=True,
                            size = 28,
                            transform=transform)
  #------ train
  train_x = train_dataset.imgs
  train_x = torch.asarray(train_x, dtype = torch.float32).permute(0,3,1,2) / 255

  train_y = train_dataset.labels
  train_y = train_y.reshape(len(train_y))
  train_y = torch.asarray(train_y)

  #------ test
  test_x = test_dataset.imgs
  test_x = torch.asarray(test_x, dtype = torch.float32).permute(0,3,1,2) / 255

  test_y = test_dataset.labels
  test_y = test_y.reshape(len(test_y))
  test_y = torch.asarray(test_y)


  #return (train_dataset, test_dataset, n_classes, n_channels)
  return (train_x, train_y, test_x, test_y, n_classes, n_channels)




def dataLoading_NIHChestXray(image_dir, 
                             csv_file,
                             img_size=(224,224),
                             max_samples_per_class=None):
    """
    Betölti a NIH Chest X-ray datasetet PyTorch Tensors-be.
    Visszatérési érték:
        train_x: torch.Tensor [N, 1, H, W]
        train_y: torch.Tensor [N]
        test_x: torch.Tensor [M, 1, H, W]
        test_y: torch.Tensor [M]
        num_class: int
        channel_size: int (1, mert grayscale)
    """


    # 1️⃣ CSV beolvasása (labels)
    df = pd.read_csv(csv_file)
    
    # Például csak az első 5 osztály (ha túl nagy dataset)
    labels_list = df['Finding Labels'].unique().tolist()
    labels_map = {label:i for i,label in enumerate(labels_list)}
    
    # 2️⃣ Képek + címkék listázása
    images = []
    targets = []

    for idx, row in df.iterrows():
        filename = row['Image Index']
        labels = row['Finding Labels'].split('|')[0]  # csak az első címke (egyszerűsítés)
        if labels not in labels_map:
            continue
        label_idx = labels_map[labels]
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue
        images.append(img_path)
        targets.append(label_idx)
    
    # max_samples_per_class használata (opcionális)
    if max_samples_per_class:
        filtered_images = []
        filtered_targets = []
        counts = {}
        for img, tgt in zip(images, targets):
            if counts.get(tgt,0) >= max_samples_per_class:
                continue
            filtered_images.append(img)
            filtered_targets.append(tgt)
            counts[tgt] = counts.get(tgt,0)+1
        images = filtered_images
        targets = filtered_targets
    
    # 3️⃣ Train-test split
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        images, targets, test_size=0.2, stratify=targets, random_state=42
    )

    # 4️⃣ Transformáció (Tensor + Normalize)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),  # [0,1]
        # transforms.Normalize(mean=[0.5], std=[0.5]) # opcionális
    ])

    def load_images(img_list):
        data = []
        for p in img_list:
            img = Image.open(p).convert('L')  # grayscale
            img = transform(img)
            data.append(img)
        return torch.stack(data)

    train_x = load_images(train_imgs)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = load_images(test_imgs)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    num_class = len(labels_map)
    channel_size = 1  # grayscale

    return train_x, train_y, test_x, test_y, num_class, channel_size


class FewShotDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
     return self.x[idx], self.y[idx]


# SAMPLES, CHANNELS, HEIGHT, WIDTH <- TRAIN_X
def traindatasetMasking(train_x, train_y, num_class, max_size_as_class):


  indexs = np.arange(train_x.shape[0])

  train_targets_index_bool = [train_y == y for y in range(num_class)]# Choosing X images from each classes as labeled

  train_targets_indexs = [np.asarray(indexs[index_bool]) for index_bool in train_targets_index_bool]

  for c in range(num_class):
      assert len(train_targets_indexs[c]) >= max_size_as_class, \
          f"Class {c} has only {len(train_targets_indexs[c])} samples, but max_size_as_class={max_size_as_class}"

  train_mask_index = np.hstack(np.array([train_targets_indexs[target][:max_size_as_class] for target in range(num_class)])) #Labeling

  train_x_filtered = train_x[train_mask_index]
  train_y_filtered = train_y[train_mask_index]

  return train_x_filtered, train_y_filtered