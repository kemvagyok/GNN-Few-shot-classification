from preprocessing.image.fewShotImageDataset import FewShotImageDataset
from preprocessing.text.fewShotTextDataSet import FewShotTextDataset

def build_dataset(datasetType, x, y, num_class, device=None):
    if datasetType == "image":
        return FewShotImageDataset(
                    x=x, 
                    y=y, 
                    num_class=num_class, 
                    device=device)
    elif datasetType == "text":
        return FewShotTextDataset(
                    x=x, 
                    y=y, 
                    num_class=num_class, 
                    device=device)
    else:
        raise ValueError(f"Unsupported dataset type: {datasetType}")
