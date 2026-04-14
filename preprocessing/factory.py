from preprocessing.image.fewShotImageDataset import FewShotImageDataset
from preprocessing.text.fewShotTextDataSet import FewShotTextDataset

def build_dataset(datasetType, train_x, train_y, num_class, val_x = None, val_y = None, device=None):
    if datasetType == "image":
        return FewShotImageDataset(
                    x_other=train_x, 
                    y_other=train_y, 
                    x_another=val_x,
                    y_another=val_y,
                    num_class=num_class, 
                    device=device)
    elif datasetType == "text":
        return FewShotTextDataset(
                    x_other=train_x, 
                    y_other=train_y, 
                    x_another=val_x,
                    y_another=val_y,
                    num_class=num_class, 
                    device=device)
    else:
        raise ValueError(f"Unsupported dataset type: {datasetType}")
