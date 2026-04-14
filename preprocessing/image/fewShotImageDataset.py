from preprocessing.base.baseFewShotDataset import BaseFewShotDataset
import torch
import numpy as np

class FewShotImageDataset(BaseFewShotDataset):
    def __init__(self, x_other, y_other, num_class, x_another=None, y_another=None, device=None):
        super().__init__(y_other, num_class, device)
        self.other_x = x_other
        self.other_y = y_other

        if x_another is not None and y_another is not None:
            self.another_x = x_another
            self.another_y = y_another
            """
            self.input_ids = torch.cat((self.other_input_ids, self.another_input_ids))
            self.attention_mask = torch.cat((self.other_attention_mask, self.another_attention_mask))
            self.y = torch.cat((self.other_y, self.another_y))
            self.train_mask = self.create_train_mask(len(self.input_ids), len(self.y))
            """ 
            self.update_train_mask()
        else:
            #Teszthalmaz
            self.x = self.other_x
            self.y = self.other_y
            self.train_mask = None
    def update_train_mask(self, max_size_as_class=None):
        if max_size_as_class is None:
            idx = np.arange(len(self.other_y))
        else:
            idx = self.traindatasetFiltering(self.other_y, self.num_class, max_size_as_class)

        train_x = self.other_x[idx]
        train_y = self.other_y[idx]

        self.x = torch.cat((train_x, self.another_x))
        self.y = torch.cat((train_y, self.another_y))

        self.train_mask = self.create_train_mask(len(train_x), len(self.y))

    def __getitem__(self, idx):
        return {
            "inputs": {
                "x": self.x[idx]
            },
            "labels": self.y[idx],
            "train_mask": self.train_mask[idx]
        }

    def __len__(self):
        return len(self.y)
    
    def get_train_val_size(self):
        return len(self.other_y), len(self.another_y)
    
    def get_all(self):
        return {
            "inputs": {
                "x": self.x
            },
            "labels": self.y,
            "train_mask": self.train_mask
        }