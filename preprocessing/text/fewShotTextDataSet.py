from preprocessing.base.baseFewShotDataset import BaseFewShotDataset

class FewShotTextDataset(BaseFewShotDataset):
    def __init__(self, x, y, num_class, device=None):
        super().__init__(y, num_class, device)

        self.input_ids = x["input_ids"]
        self.attention_mask = x["attention_mask"]

    def __getitem__(self, idx):
        return {
            "inputs": {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            },
            "labels": self.y[idx]
        }

    def __len__(self):
        return len(self.y)
    
    def get_all(self):
        return {
            "inputs": {
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask
            },
            "labels": self.y,
        }