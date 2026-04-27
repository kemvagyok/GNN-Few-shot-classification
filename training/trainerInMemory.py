import torch
import faiss
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import wandb

class TrainerInMemory:
    def __init__(self, embedder, gnn, criterion, config, device, metric_fn=None):
        self.embedder = embedder.to(device)
        self.gnn = gnn.to(device)

        self.criterion = criterion.to(device)
        self.metric_fn = metric_fn

        self.config = config
        self.device = device

        self.opt_embedder = torch.optim.Adam(
            self.embedder.parameters(),
            lr=config.lr_embedder
        )
        self.opt_gnn = torch.optim.Adam(
            self.gnn.parameters(),
            lr=config.lr_gcn
        )

    # ---------------------------
    # TRAIN ENTRY POINT
    # ---------------------------
    def train(self, train_dataset, val_dataset=None, test_dataset=None, K_hop=5):
        best_val = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

        for epoch in range(self.config.epochs_max):

            train_loss = self._train_epoch(
                train_loader,
                val_loader,
                K_hop,
                epoch
            )

            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader)

                print(
                    f"[Epoch {epoch}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Metric: {val_metric:.4f}"
                )
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metric": val_metric
                }, step=epoch)

                if val_metric > best_val:
                    best_val = val_metric

            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
                wandb.log({"train_loss" : train_loss}, step=epoch)
        if test_loader is not None:
            test_loss, test_metric = self.evaluate(test_loader)
            print(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f}")
            wandb.log({"test_loss" : test_loss, "test_metric": test_metric})
        return best_val

    # ---------------------------
    # TRAIN EPOCH
    # ---------------------------
    def _train_epoch(self, train_loader, val_loader, K_hop, epoch):

        self.embedder.train()
        self.gnn.train()

        embeddings_list = []
        labels_list = []

        train_size = 0

        # ---------------------------
        # EMBEDDINGS: TRAIN + VAL
        # ---------------------------
        with torch.no_grad():

            # TRAIN
            for batch in train_loader:
                x = self._to_device(batch["inputs"])
                emb = self.embedder(**x)    
                embeddings_list.append(emb)
                labels_list.append(batch["labels"].to(self.device))

                train_size += emb.size(0)

            # VAL (csak embedding)
            if val_loader is not None:
                for batch in val_loader:
                    x = self._to_device(batch["inputs"])
                    emb = self.embedder(**x)

                    embeddings_list.append(emb)
                    labels_list.append(batch["labels"].to(self.device))

        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        # ---------------------------
        # GRAPH BUILD (train + val nodes)
        # ---------------------------
        graph = self._build_graph(
            embeddings,
            labels,
            self.config.K_neigh
        )

        train_mask = torch.zeros(len(embeddings), dtype=torch.bool, device=self.device)
        train_mask[:train_size] = True
        graph.train_mask = train_mask

        # ---------------------------
        # GNN TRAINING
        # ---------------------------
        loader = NeighborLoader(
            graph,
            num_neighbors=[self.config.K_neigh] * K_hop,
            input_nodes=train_mask,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        total_loss = 0

        for subgraph in loader:

            self.opt_embedder.zero_grad()
            self.opt_gnn.zero_grad()

            out = self.gnn(subgraph)
            global_node_ids = subgraph.n_id
            train_nodes_in_batch = graph.train_mask[global_node_ids]
            mask = train_nodes_in_batch
            loss = self.criterion(
                out[mask],
                subgraph.y[mask]
            )

            loss.backward()

            self.opt_embedder.step()
            self.opt_gnn.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ---------------------------
    # GRAPH BUILDER (FAISS)
    # ---------------------------
    def _build_graph(self, embeddings, labels, K):

        emb = embeddings.detach().cpu().numpy()

        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)

        _, I = index.search(emb, K + 1)

        neighbors = torch.tensor(I[:, 1:], device=self.device)
        edge_index = self._to_edge_index(neighbors)

        return Data(
            x=embeddings,
            edge_index=edge_index,
            y=labels
        )

    def _to_edge_index(self, neighbors):
        src = torch.arange(neighbors.size(0), device=self.device).repeat_interleave(neighbors.size(1))
        dst = neighbors.flatten()
        return torch.stack([src, dst], dim=0)

    # ---------------------------
    # EVALUATION
    # ---------------------------
    @torch.no_grad()
    def evaluate(self, loader):

        self.embedder.eval()
        self.gnn.eval()

        emb_list = []
        label_list = []

        for batch in loader:
            x = self._to_device(batch["inputs"])
            emb = self.embedder(**x)

            emb_list.append(emb)
            label_list.append(batch["labels"].to(self.device))

        embeddings = torch.cat(emb_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        graph = self._build_graph(
            embeddings,
            labels,
            self.config.K_neigh
        )

        out = self.gnn(graph)

        loss = self.criterion(out, labels)
        preds = out.argmax(dim=1)
        if self.metric_fn:
            metric = self.metric_fn(preds.to('cpu'), labels.to('cpu'), loader.dataset.base.class_num)
        else:
            metric = (preds == labels).float().mean().item()

        return loss.item(), metric

    # ---------------------------
    # DEVICE HELPER
    # ---------------------------
    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}