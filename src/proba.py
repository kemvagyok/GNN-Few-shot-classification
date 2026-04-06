import torch
import torch.nn as nn
from torchvision import models
from transformers import BertTokenizer, BertModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"

encoding = tokenizer.encode(text)

print("Encoded text:", encoding)

tokens = tokenizer.convert_ids_to_tokens(encoding)

print("Tokens:", tokens)

bert = BertModel.from_pretrained("bert-base-uncased")

input_ids = torch.tensor([encoding])  # Batch size of 1
attention_mask = torch.ones_like(input_ids)  # Attention mask (all ones for simplicity)

outputs = bert(input_ids=input_ids, attention_mask=attention_mask)

cls = outputs.last_hidden_state[:, 0, :]  # [CLS]

proj = nn.Linear(768, 64) 

out = proj(cls)  # [batch, 64]

print("CLS token representation shape:", out.shape)

print("CLS token representation:", out)

import matplotlib.pyplot as plt
import seaborn as sns

attentions = outputs.attentions  # List of attention matrices for each layer and head

# pl. első layer, első head
attn = attentions[0][0][0].detach().numpy()

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

plt.figure(figsize=(8, 6))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")

plt.title("Attention Heatmap (Layer 0, Head 0)")
plt.xlabel("Key (attended to)")
plt.ylabel("Query (attending)")
plt.savefig("attention.png")
