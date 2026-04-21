from preprocessing.base.baseFewShotDataset import BaseFewShotDataset

class FewShotImageDataset(BaseFewShotDataset):
    def __init__(self, x, y, num_class, device=None):
        super().__init__(y, num_class, device)
        self.x = x

    def __getitem__(self, idx):
        return {
            "inputs": {
                "x": self.x[idx]
            },
            "labels": self.y[idx]
        }

    def __len__(self):
        return len(self.y)
    
    def get_all(self):
        return {
            "inputs": {
                "x": self.x
            },
            "labels": self.y,
        }