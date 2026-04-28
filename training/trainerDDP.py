import time
import wandb
import torch
import torch.distributed as dist
from torch_geometric.loader import NeighborLoader

class TrainerDDP:
    def __init__(self, embedder, gnn, graph_builder,  criterion,
                 config, device, local_rank=0, metric_fn=None):

        self.embedder = embedder.to(device)
        self.graph_builder = graph_builder
        self.gnn = gnn.to(device)

        self.config = config
        self.device = device
        self.rank = local_rank
        self.world_size = dist.get_world_size()

        self.criterion = criterion.to(device)
        self.metric_fn = metric_fn

        self.opt_embedder = torch.optim.Adam(
            self.embedder.parameters(), lr=config.lr_embedder)
        self.opt_gnn = torch.optim.Adam(
            self.gnn.parameters(), lr=config.lr_gcn
        )

    def _is_main(self):
        return self.rank == 0

    def _reduce_mean(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor

    def train(self, train_dataset, val_dataset=None, test_dataset=None, K_hop=None):
        best_val_acc = 0

        for epoch in range(self.config.epochs_max):
            train_loss = self._train_epoch(
                labeled_dataset = train_dataset, 
                unlabeled_dataset = val_dataset,
                K_hop = K_hop)

            if val_dataset is not None:
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

        if test_dataset is not None:
            test_loss, test_metric = self.evaluate(test_dataset)

            print(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f}")
            wandb.log({
                "test_loss": test_loss,
                "test_metric": test_metric
            })   

        return best_val_acc

    def _train_epoch(self, labeled_dataset, unlabeled_dataset, K_hop):
        self.embedder.train()
        self.gnn.train()

        data_labeled = labeled_dataset.get_all()
        data_unlabeled   = unlabeled_dataset.get_all()
        inputs, y, train_idx, val_idx = self._merge_datasets(data_labeled, data_unlabeled)

        start = time.time()
        total_loss = 0
            # ---- graph build ----
        with torch.no_grad():
            embeddings_for_graph = self._embed_chunked(inputs, chunk_size=256)

            t1 = time.time()
            fullGraph = self.graph_builder(
                latens=embeddings_for_graph,
                train_val_y=y,
                K_neigh=self.config.K_neigh,
                device=self.device,
            )

            del embeddings_for_graph
            torch.cuda.empty_cache()

        train_mask = torch.zeros(fullGraph.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        fullGraph.train_mask = train_mask

        loader = NeighborLoader(
            fullGraph,
            num_neighbors=[self.config.K_neigh] * K_hop,
            input_nodes=train_idx,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        for subgraph in loader:
            batch_inputs = {
                k: v[subgraph.n_id].to(self.device)
                for k, v in inputs.items()
            }

            loss = self._step(subgraph, raw_inputs=batch_inputs)
            total_loss += loss.item()
                
            del batch_inputs
            torch.cuda.empty_cache()

        loss = torch.tensor(total_loss / len(loader), device=self.device)
        loss = self._reduce_mean(loss)

        return loss.item()

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
            metric_value = self.metric_fn(preds, y, dataset.base.class_num)
        else:
            metric_value = (preds == y).float().mean()

        # 🔑 DDP sync
        loss = self._reduce_mean(loss)
        metric_value = self._reduce_mean(metric_value)

        return loss.item(), metric_value.item()

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
    
    def _merge_datasets(self, labeled_data, unlabeled_data):
        labeled_inputs = labeled_data["inputs"]
        unlabeled_inputs = unlabeled_data["inputs"]

        merged_inputs = {
            k: torch.cat([labeled_inputs[k].to(self.device), unlabeled_inputs[k].to(self.device)], dim=0)
            for k in labeled_inputs
        }

        labeled_y = labeled_data["labels"]
        unlabeled_y = unlabeled_data["labels"]

        merged_y = torch.cat([labeled_y, unlabeled_y], dim=0)

        train_size = labeled_y.size(0)

        train_idx = torch.arange(train_size)
        val_idx = torch.arange(train_size, merged_y.size(0))

        return merged_inputs, merged_y, train_idx, val_idx