import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

print(data)

train_idx = data.train_mask.nonzero(as_tuple=True)[0]
val_idx = data.val_mask.nonzero(as_tuple=True)[0]

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x
    
from torch_geometric.loader import NeighborLoader

def train_one_epoch(model, loader, optimizer, data):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)

        loss = F.cross_entropy(
            out[:batch.batch_size],
            data.y[batch.n_id[:batch.batch_size]]
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, data):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)

        pred = out[:batch.batch_size].argmax(dim=1)
        y = data.y[batch.n_id[:batch.batch_size]]

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total

def run_experiment(data, train_idx, val_idx, k_values):
    results = {}

    for k in k_values:
        print(f"\n===== k-hop = {k} =====")

        num_neighbors = [10] * k

        train_loader = NeighborLoader(
            data,
            input_nodes=train_idx,
            num_neighbors=num_neighbors,
            batch_size=64,
            shuffle=True,
        )

        val_loader = NeighborLoader(
            data,
            input_nodes=val_idx,
            num_neighbors=num_neighbors,
            batch_size=64,
            shuffle=False,
        )

        model = GNN(
            in_channels=data.num_features,
            hidden_channels=64,
            out_channels=dataset.num_classes,
            num_layers=len(k_values) - 1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_acc = 0

        for epoch in range(1, 16):
            loss = train_one_epoch(model, train_loader, optimizer, data)
            acc = evaluate(model, val_loader, data)

            best_acc = max(best_acc, acc)

            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

        results[k] = best_acc

    return results


k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results = run_experiment(data, train_idx, val_idx, k_values)

print("\n=== FINAL RESULTS ===")
for k, acc in results.items():
    print(f"k={k} → best val acc = {acc:.4f}")