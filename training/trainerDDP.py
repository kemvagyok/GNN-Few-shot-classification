import time

import wandb
import torch
import torch.distributed as dist
from torch_geometric.loader import NeighborLoader


class TrainerDDP:
    def __init__(self, embedder, graph_builder, gnn, criterion,
                 config, device, rank=0, world_size=1, metric_fn=None):

        self.embedder = embedder.to(device)
        self.graph_builder = graph_builder
        self.gnn = gnn.to(device)

        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.criterion = criterion.to(device)
        self.metric_fn = metric_fn

        self.opt_embedder = torch.optim.Adam(
            self.embedder.parameters(), lr=config.lr_embedder)
        self.opt_gnn = torch.optim.Adam(
            self.gnn.parameters(), lr=config.lr_gcn
        )

    # ---------------------------
    # 🔧 DDP utils
    # ---------------------------
    def _is_main(self):
        return self.rank == 0

    def _reduce_mean(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor

    # ---------------------------
    # 🚀 TRAIN
    # ---------------------------
    def train(self, train_dataset, val_dataset=None, K_hop=None):
        best_val_acc = 0

        for epoch in range(self.config.epochs_max):
            train_loss = self._train_epoch(train_dataset, K_hop)

            if val_dataset is not None:
                # ❗ FIX: csak egyszer evaluate
                val_loss, val_metric = self.evaluate(val_dataset)

                if self._is_main():
                    print(f"[Epoch {epoch}] "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Metric: {val_metric:.4f}")

                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_metric": val_metric
                    }, step=epoch)

                if val_metric > best_val_acc:
                    best_val_acc = val_metric
            else:
                if self._is_main():
                    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        return best_val_acc

    # ---------------------------
    # 🔥 TRAIN EPOCH
    # ---------------------------
    def _train_epoch(self, dataset, K_hop):
        self.embedder.train()
        self.gnn.train()

        data = dataset.get_all()
        inputs = data["inputs"]
        y = data["labels"]
        train_mask = data["train_mask"]
        start = time.time()
        total_loss = 0
            # ---- graph build ----
        with torch.no_grad():
            t0 = time.time()
            embeddings_for_graph = self._embed_chunked(inputs, chunk_size=256)

            if self._is_main():
                print(f"Embedding time: {time.time() - t0:.2f}s")

            t1 = time.time()
            fullGraph = self.graph_builder(
                latens=embeddings_for_graph,
                train_val_y=y,
                train_mask=train_mask,
                K_neigh=self.config.K_neigh,
                device=self.device,
            )
            if self._is_main():
                print(f"Graph build time: {time.time() - t1:.2f}s")
            del embeddings_for_graph
            torch.cuda.empty_cache()

        # 🔑 DDP shard
        train_idx = fullGraph.train_mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.chunk(self.world_size)[self.rank]

        loader = NeighborLoader(
            fullGraph,
            num_neighbors=[self.config.K_neigh] * K_hop,
            input_nodes=train_idx,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        t2 = time.time()
        for subgraph in loader:
            batch_inputs = {
                k: v[subgraph.n_id].to(self.device)
                for k, v in inputs.items()
            }

            loss = self._step(subgraph, raw_inputs=batch_inputs)
            total_loss += loss.item()

            if self._is_main():
                print(f"GNN training time: {time.time() - t2:.3f}s")
                print(f"Total epoch time: {time.time() - start:.3f}s")
                
            del batch_inputs
            torch.cuda.empty_cache()

        loss = torch.tensor(total_loss / len(loader), device=self.device)
        loss = self._reduce_mean(loss)

        return loss.item()

    # ---------------------------
    # ⚡ STEP
    # ---------------------------
    def _step(self, graph, raw_inputs=None):
        self.opt_embedder.zero_grad()
        self.opt_gnn.zero_grad()

        if raw_inputs is not None:
            with torch.no_grad():
                graph.x = self.embedder(**raw_inputs)

        outputs = self.gnn(graph)

        loss = self.criterion(
            outputs[graph.train_mask],
            graph.y[graph.train_mask]
        )

        loss.backward()

        self.opt_embedder.step()
        self.opt_gnn.step()

        return loss

    # ---------------------------
    # 📉 EVAL
    # ---------------------------
    @torch.no_grad()
    def evaluate(self, dataset):
        self.embedder.eval()
        self.gnn.eval()

        data = dataset.get_all()
        inputs = data["inputs"]
        y = data["labels"].to(self.device)

        embeddings = self._embed_chunked(inputs, chunk_size=256).to(self.device)

        graph = self.graph_builder(
            embeddings, y, None,
            self.config.K_neigh,
            device=self.device
        )

        outputs = self.gnn(graph)

        loss = self.criterion(outputs, y)

        preds = torch.argmax(outputs, dim=1)

        if self.metric_fn is not None:
            metric_value = self.metric_fn(preds, y)
        else:
            metric_value = (preds == y).float().mean()

        # 🔑 DDP sync
        loss = self._reduce_mean(loss)
        metric_value = self._reduce_mean(metric_value)

        return loss.item(), metric_value.item()

    # ---------------------------
    # 🧠 EMBEDDING
    # ---------------------------
    @torch.no_grad()
    def _embed_chunked(self, raw_inputs, chunk_size=256):
        chunks = []
        n = next(iter(raw_inputs.values())).shape[0]

        for i in range(0, n, chunk_size):
            chunk = {
                k: v[i:i + chunk_size].to(self.device)
                for k, v in raw_inputs.items()
            }
            chunks.append(self.embedder(**chunk).cpu())
            del chunk

        return torch.cat(chunks, dim=0)