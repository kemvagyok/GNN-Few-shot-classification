import wandb
import torch
from torch_geometric.loader import NeighborLoader

class Trainer:
    def __init__(self, embedder, graph_builder, gnn, criterion, config, device, metric_fn=None):
        self.embedder = embedder.to(device)
        self.graph_builder = graph_builder
        self.gnn = gnn.to(device)
        self.config = config
        self.device = device
        self.criterion = criterion.to(device)
        self.metric_fn = metric_fn
        self.opt_embedder = torch.optim.Adam(
            self.embedder.parameters(), lr=config.lr_embedder)
        self.opt_gnn = torch.optim.Adam(
            self.gnn.parameters(), lr=config.lr_gcn
        )

    def train(self, train_dataset, val_dataset=None, K_hop=None):
        best_val_acc = 0
        for epoch in range(self.config.epochs_max):
            train_loss = self._train_epoch(train_dataset, K_hop)
            if val_dataset is not None:
                val_loss, val_acc = self.evaluate(val_dataset)

                val_loss, val_metric = self.evaluate(val_dataset)

                print(f"[Epoch {epoch}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Metric: {val_metric:.4f}")
                
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metric": val_metric
                }, step=epoch)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
        return best_val_acc

    def _train_epoch(self, dataset, K_hop):
        self.embedder.train()
        self.gnn.train()

        data = dataset.get_all()

        inputs = data["inputs"]
        y = data["labels"]
        train_mask = data["train_mask"]

        total_loss = 0
        #MINIBATCH TRAINING: build the full graph once, then sample many small subgraphs for training
        if self.config.use_minibatch:
            # Build graph topology only — embeddings stay on CPU, freed immediately after
            with torch.no_grad():
                embeddings_for_graph = self._embed_chunked(inputs, chunk_size=256)
                fullGraph = self.graph_builder(
                    latens=embeddings_for_graph,
                    train_val_y=y,
                    train_mask=train_mask,
                    K_neigh=self.config.K_neigh,
                    device=self.device,
                )
                del embeddings_for_graph   # free before the training loop starts
                torch.cuda.empty_cache()

            train_idx = fullGraph.train_mask.nonzero(as_tuple=False).view(-1)
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

            return total_loss / len(loader)
        #FULL-BATCH TRAINING: embed all nodes, build one giant graph, and train on it repeatedly
        else:
            embeddings = self.embedder(**inputs)
            fullGraph = self.graph_builder(
                latens=embeddings,
                train_val_y=y,
                train_mask=train_mask,
                K_neigh=self.config.K_neigh,
                device=self.device,
            )
            loss = self._step(fullGraph)
            total_loss += loss.item()
            return total_loss

    @torch.no_grad()
    def _embed_chunked(self, raw_inputs, chunk_size=256):
        """Embed all nodes in small chunks to avoid one giant GPU allocation.
        Returns a CPU tensor — move to device only when actually needed."""
        chunks = []
        n = next(iter(raw_inputs.values())).shape[0]
        for i in range(0, n, chunk_size):
            chunk = {k: v[i : i + chunk_size].to(self.device) for k, v in raw_inputs.items()}
            chunks.append(self.embedder(**chunk).cpu())
            del chunk
        return torch.cat(chunks, dim=0)

    def _step(self, graph, raw_inputs=None):
        self.opt_embedder.zero_grad()
        self.opt_gnn.zero_grad()

        if raw_inputs is not None:
            # No clone() needed — NeighborLoader returns a fresh Data object each iter
            graph.x = self.embedder(**raw_inputs)
        outputs = self.gnn(graph)
        loss = self.criterion(outputs[graph.train_mask], graph.y[graph.train_mask])
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
        y = data["labels"]
        y = y.to(self.device)
        embeddings = self._embed_chunked(inputs, chunk_size=256).to(self.device)
        outputs = self.gnn(
            self.graph_builder(embeddings, y, None, self.config.K_neigh, device=self.device)
        )
        print(outputs.device, y.device)
        loss = self.criterion(outputs, y).item()
        preds = torch.argmax(outputs, dim=1)
        
        if self.metric_fn is not None:
            metric_value = self.metric_fn(preds, y)
        else:
            metric_value = (preds == y).sum().item() / y.size(0)

        return loss, metric_value