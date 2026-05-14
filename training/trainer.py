import time
import wandb
import torch
from torch_geometric.loader import NeighborLoader
from utils import EarlyStopping
#--------------
from preprocessing.indexedDataset import IndexedDataset
#--------------

class Trainer:
    def __init__(self, embedder, graph_builder, gnn, criterion, config, device, metric_fn=None):

        self.embedder = embedder.to(device)
        self.gnn = gnn.to(device)
        self.graph_builder = graph_builder

        self.config = config
        self.device = device

        self.criterion = criterion.to(device)
        self.metric_fn = metric_fn

        self.opt_embedder = torch.optim.Adam(
            self.embedder.parameters(), lr=config.lr_embedder)
        self.opt_gnn = torch.optim.Adam(
            self.gnn.parameters(), lr=config.lr_gcn
        )

    def train(self, train_dataset, val_dataset=None, test_dataset=None, K_hop=None, **kwargs):

        best_val_metric = 0

        stopper = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.delta
        )

        for epoch in range(self.config.epochs_max):

            train_loss, train_metric = self._train_epoch(
                labeled_dataset = train_dataset, 
                unlabeled_dataset = val_dataset,
                epoch = epoch,
                K_hop = K_hop
                )

            if val_dataset is not None:
                
                val_loss, val_metric = self.evaluate(val_dataset)

                print(f"[Epoch {epoch}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Metric: {train_metric:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Metric: {val_metric:.4f}")
                
                wandb.log({
                    "train_loss": train_loss,
                    "train_metric": train_metric,
                    "val_loss": val_loss,
                    "val_metric": val_metric
                }, step=epoch)

                best_val_metric = max(best_val_metric, val_metric)

                if stopper.early_stop(val_loss): #Korai leállás

                    print(f"Early stopping! Epoch: {epoch}")

                    break
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        if test_dataset is not None:

            test_loss, test_metric = self.evaluate(test_dataset)

            print(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f}")
            wandb.log({
                "test_loss": test_loss,
                "test_metric": test_metric
            })   

        return best_val_metric

    def _train_epoch(self, labeled_dataset: IndexedDataset, unlabeled_dataset: IndexedDataset, epoch: int, K_hop: int|None):
        
        self.embedder.train()
        self.gnn.train()

        data_labeled = labeled_dataset.get_all()
        data_unlabeled   = unlabeled_dataset.get_all()

        inputs, y, train_idx, _ = self._merge_datasets(
            data_labeled, data_unlabeled
            )

        # Build graph topology only — embeddings stay on CPU, freed immediately after
        with torch.no_grad():

            embeddings_for_graph = self._embed_chunked(inputs)

        # =========================
        # GRAPH BUILD (EPOCH-LEVEL)
        # =========================
        start = time.time()
        if (epoch < 20) or (epoch % self.config.graph_refresh == 0):
            self.graph = self.graph_builder(
                latens=embeddings_for_graph,
                y=y,
                K_neigh=self.config.K_neigh,
                device=self.device,
            )
        print("A", time.time()-start)

        #-----------------------
        train_mask = torch.zeros(
            self.graph.num_nodes, dtype=torch.bool, device=self.device
        )
        train_mask[train_idx] = True
        self.graph.train_mask = train_mask
        #-----------------------

        start = time.time()
        loader = NeighborLoader(
            self.graph,
            num_neighbors=[self.config.K_neigh] * K_hop,
            input_nodes=train_idx,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )
        print("B", time.time()-start)

        total_loss = 0
        total_metric_value = 0

        start = time.time()
        for subgraph in loader:

            n_id = subgraph.n_id

            subgraph = subgraph.to(self.device)

            loss, metric_value  = self._step(
                graph = subgraph, 
                n_id=n_id,
                inputs = inputs, 
                class_num = labeled_dataset.base.class_num
                )
                
            total_loss += loss.item()
            total_metric_value += metric_value

        print("C", time.time()-start)

        return total_loss / len(loader), total_metric_value / len(loader)

    # =========================
    # STEP
    # =========================
    def _step(self, graph, n_id, class_num, inputs=None):

        self.opt_embedder.zero_grad()
        self.opt_gnn.zero_grad()

        batch_inputs = {
            k: v[n_id].to(
                self.device, 
                non_blocking=True
                )
            for k, v in inputs.items()
        }
        # =========================
        # TRAINABLE EMBEDDER
        # =========================
        graph.x = self.embedder(**batch_inputs)

        outputs = self.gnn(graph)

        loss = self.criterion(
            outputs[graph.train_mask], 
            graph.y[graph.train_mask]
            )

        if self.metric_fn is not None:
            metric = self.metric_fn(
                outputs[graph.train_mask], 
                graph.y[graph.train_mask], 
                num_classes = class_num
                )
        else:
            metric = (outputs[graph.train_mask] == graph.y[graph.train_mask]
            ).sum().item() / graph.y[graph.train_mask].size(0)

        loss.backward()

        self.opt_embedder.step()
        self.opt_gnn.step()

        return loss, metric

    # =========================
    # EVAL
    # =========================
    @torch.no_grad()
    def evaluate(self, dataset):

        self.embedder.eval()
        self.gnn.eval()

        data = dataset.get_all()
        inputs = {k: v.to(self.device) for k, v in data["inputs"].items()}
        y = data["labels"].to(self.device)
        
        embeddings = self._embed_chunked(inputs, chunk_size=256).to(self.device)

        graph = self.graph_builder(
            latens=embeddings,
            y=y,
            K_neigh=self.config.K_neigh,
            device=self.device
        )

        outputs = self.gnn(graph)
        
        loss = self.criterion(outputs, y)
        
        preds = torch.argmax(outputs, dim=1)
        
        if self.metric_fn is not None:
            metric = self.metric_fn(preds, y, num_classes = dataset.base.class_num)
        else:
            metric = (preds == y).sum().item() / y.size(0)

        return loss, metric

    # =========================
    # EMBEDDING (TRAINABLE)
    # =========================
    @torch.no_grad()
    def _embed_chunked(self, raw_inputs, chunk_size=256):

        chunks = []
        n = next(iter(raw_inputs.values())).shape[0]
        
        for i in range(0, n, chunk_size):

            chunk = {
                k: v[i : i + chunk_size].to(self.device, non_blocking=True)
                for k, v in raw_inputs.items()
                }

            chunks.append(self.embedder(**chunk).cpu())

        return torch.cat(chunks, dim=0)

    # =========================
    # MERGE
    # =========================
    def _merge_datasets(self, labeled_data, unlabeled_data):

        labeled_inputs = labeled_data["inputs"]
        unlabeled_inputs = unlabeled_data["inputs"]

        merged_inputs = {
            k: torch.cat([labeled_inputs[k],
                        unlabeled_inputs[k]], dim=0)
            for k in labeled_inputs
        }

        labeled_y = labeled_data["labels"]
        unlabeled_y = unlabeled_data["labels"]

        merged_y = torch.cat([
            labeled_y, unlabeled_y
            ], dim=0)

        train_size = labeled_y.size(0)

        train_idx = torch.arange(train_size)
        val_idx = torch.arange(train_size, merged_y.size(0))

        return merged_inputs, merged_y, train_idx, val_idx