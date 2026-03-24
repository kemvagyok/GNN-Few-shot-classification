import torch
import torch.nn as nn
import torch.nn.functional as F

class textcnnModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, laten, padding_idx=0):
        super().__init__()
        # 1. Embedding layer (token → vector)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # 2. Két konvolúciós réteg
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=4, padding=2)

        # 3. Output layer
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        """
        x: (batch_size, seq_len) -> token id-k
        """
        # Embedding → (B, L, E)
        x = self.embedding(x)
        # Transpose a Conv1D-hez → (B, E, L)
        x = x.transpose(1, 2)

        # Két konvolúciós réteg
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Global max pooling → (B, C)
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)

        # Output
        return self.fc(x)