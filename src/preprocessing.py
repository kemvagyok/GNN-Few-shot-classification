from torchvision import datasets
from torchvision.transforms import ToTensor

def MNIST_data_loading():
  transform = ToTensor()
  train_data = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)
  test_data = datasets.MNIST('./data',
                                train=False,
                                download=True,
                                transform=transform)
  return (train_data, test_data)