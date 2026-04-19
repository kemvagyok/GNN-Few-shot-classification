from preprocessing.base.baseFewShotDataset import BaseFewShotDataset
import torch
import numpy as np

class FewShotTextDataset(BaseFewShotDataset):
    def __init__(self, x_other, y_other, num_class, x_another = None, y_another = None,  device=None):
        super().__init__(y_other, num_class, device)
        # unpack dict formátumból
        self.other_input_ids = x_other["input_ids"]
        self.other_attention_mask = x_other["attention_mask"]
        self.other_y = y_other

        if x_another is not None and y_another is not None:
            self.another_input_ids = x_another["input_ids"]
            self.another_attention_mask = x_another["attention_mask"]
            self.another_y = y_another
            self.update_train_mask()
        else:
            #Teszthalmaz
            self.input_ids = self.other_input_ids
            self.attention_mask = self.other_attention_mask
            self.y = self.other_y
            self.train_mask = None

    def update_train_mask(self, max_size_as_class=None):
        if max_size_as_class is None:
            idx = np.arange(len(self.other_y))
        else:
            idx = self.traindatasetFiltering(self.other_y, self.num_class, max_size_as_class)

        train_ids = self.other_input_ids[idx]
        train_att = self.other_attention_mask[idx]
        train_y = self.other_y[idx]

        self.input_ids = torch.cat((train_ids, self.another_input_ids))
        self.attention_mask = torch.cat((train_att, self.another_attention_mask))
        self.y = torch.cat((train_y, self.another_y))
        self.train_mask = self.create_train_mask(len(train_ids), len(self.y))

    def __getitem__(self, idx):
        return {
            "inputs": {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            },
            "labels": self.y[idx],
            "train_mask": self.train_mask[idx]
        }

    def __len__(self):
        return len(self.other_input_ids)
    
    def get_train_val_size(self):
        return len(self.other_input_ids), len(self.another_input_ids)
   
    def get_all(self):
        return {
            "inputs": {
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask
            },
            "labels": self.y,
            "train_mask": self.train_mask
        }