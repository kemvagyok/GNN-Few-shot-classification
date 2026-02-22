import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, INFO, Evaluator

def dataLoading_MNIST():
  transform = ToTensor()
  train_dataset = datasets.MNIST('/root',
                              train=True,
                              download=True,
                              transform=transform)
  test_dataset = datasets.MNIST('/root',
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
  #return (train_dataset, test_dataset, n_classes, n_channels)
  return (train_x, train_y, test_x, test_y, n_classes, n_channels)


def dataLoading_medMNIST():
  data_flag = 'dermamnist'
  download = True

  info = INFO[data_flag]
  #task = info['task']

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