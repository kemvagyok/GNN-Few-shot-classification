import wandb
import torch
import torch.distributed as dist

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from utils import EarlyStopping
#--------------
from preprocessing.indexedDataset import IndexedDataset
#--------------
from torch.nn.parallel import DistributedDataParallel as DDP
#--------------
import time

class TrainerDDP:
    def __init__(self, embedder, gnn, graph_builder, criterion,
                 config, device, local_rank=0, metric_fn=None):

        self.device = device
        self.rank = local_rank
        self.world_size = dist.get_world_size()

        self.config = config
        self.metric_fn = metric_fn
        self.graph_builder = graph_builder

        # =========================
        # MODELS (DDP)
        # =========================
        self.embedder = DDP(
            embedder.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

        self.gnn = DDP(
            gnn.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )


        self.criterion = criterion.to(device)

        # =========================
        # OPTIMIZERS
        # =========================
        self.opt_embedder = torch.optim.Adam(
            self.embedder.parameters(),
            lr=config.lr_embedder
        )

        self.opt_gnn = torch.optim.Adam(
            self.gnn.parameters(),
            lr=config.lr_gcn
        )

        self.graph = None

    # =========================
    # UTILS
    # =========================
    def _is_main(self):
        return self.rank == 0

    def _reduce_mean(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor

    def _split_idx(self, idx):
        return idx[self.rank::self.world_size]

    # =========================
    # TRAIN
    # =========================
    def train(self, train_dataset, val_dataset=None, test_dataset=None, K_hop=None):

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

                if self._is_main():
                    print(
                        f"[Epoch {epoch}] "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Metric: {train_metric:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Metric: {val_metric:.4f}"
                    )

                    wandb.log({
                        "train_loss": train_loss,
                        "train_metric": train_metric,
                        "val_loss": val_loss,
                        "val_metric": val_metric
                    }, step=epoch)

                best_val_metric = max(best_val_metric, val_metric)

                if stopper.early_stop(val_loss):
                    if self._is_main():
                        print("Early stopping")

                    break

        if test_dataset is not None:

                test_loss, test_metric = self.evaluate(test_dataset)

                if self._is_main():

                    print(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f}")
                    wandb.log({
                        "test_loss": test_loss,
                        "test_metric": test_metric
                    })

        return best_val_metric

    # =========================
    # TRAIN EPOCH
    # =========================
    def _train_epoch(self, labeled_dataset: IndexedDataset, unlabeled_dataset: IndexedDataset, epoch: int, K_hop: int|None):

        self.embedder.train()
        self.gnn.train()

        #CPU-n az adatok (x, és y egyaránt)
        data_labeled = labeled_dataset.get_all()
        data_unlabeled = unlabeled_dataset.get_all()

        inputs, y, train_idx, _ = self._merge_datasets(
            data_labeled, data_unlabeled
        )

        with torch.no_grad():

            #CPU->GPU->CPU
            embeddings = self._embed_chunked(inputs)

        # =========================
        # GRAPH BUILD (EPOCH-LEVEL)
        # =========================
        start = time.time()
        if (epoch < 20) or (epoch % self.config.graph_refresh == 0) or (self.graph is None):
            #graph a GPU-n
            self.graph = self.graph_builder(
                latens=embeddings,
                y=y,
                K_neigh=self.config.K_neigh,
                device=self.device
            )
        print("A", time.time()-start, f"RANK: {self.rank}")
        #-----------------------
        train_idx = self._split_idx(train_idx)
        self.graph.train_mask = torch.zeros(
            self.graph.num_nodes, 
            dtype=torch.bool
        )
        self.graph.train_mask[train_idx] = True
        #-----------------------

        start = time.time()
        loader = NeighborLoader(
            self.graph,
            num_neighbors=[self.config.K_neigh] * K_hop,
            input_nodes=train_idx,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=False,
            pin_memory=True
        )
        print("B", time.time()-start, f"RANK: {self.rank}")
        print("train nodes:", len(train_idx))
        print("batch size:", loader.batch_size)
        print("loader len:", len(loader))
        total_loss = 0
        total_metric = 0

        
        start = time.time()
        for subgraph in loader:

            n_id = subgraph.n_id
            print(n_id)

            subgraph = subgraph.to(
                self.device,
                non_blocking=True
                )

            loss, metric  = self._step(
                graph = subgraph, 
                n_id=n_id,
                inputs = inputs, 
                class_num = labeled_dataset.base.class_num
                )

            total_loss += loss.item()
            total_metric += metric

        print("C", time.time()-start)

        loss = torch.tensor(
            total_loss / len(loader), 
            device=self.device)

        metric = torch.tensor(
            total_metric / len(loader), 
            device=self.device)


        #FOR DDP
        loss = self._reduce_mean(loss)
        metric = self._reduce_mean(metric)

        return loss.item(), metric.item()

    # =========================
    # STEP
    # =========================
    def _step(self, graph, n_id, inputs, class_num):

        self.opt_embedder.zero_grad(set_to_none = True)
        self.opt_gnn.zero_grad(set_to_none = True)

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
            metric = (
                outputs[graph.train_mask].argmax(dim=1)== graph.y[graph.train_mask]
                ).float().mean().item()

        loss.backward()

        self.opt_embedder.step()
        self.opt_gnn.step()

        return loss.detach(), metric

    # =========================
    # EVAL
    # =========================
    @torch.no_grad()
    def evaluate(self, dataset):

        self.embedder.eval()
        self.gnn.eval()

        data = dataset.get_all()
        inputs = {k: v.to(self.device) for k, v in data["inputs"].items()}
        inputs = data["inputs"]
        y = data["labels"].to(self.device)

        embeddings = self._embed_chunked(inputs)

        graph = self.graph_builder(
            latens=embeddings,
            y=y,
            K_neigh=self.config.K_neigh,
            device=self.device
        )

        outputs = self.gnn(graph)

        loss = self.criterion(outputs, y)
        preds = outputs.argmax(dim=1)

        if self.metric_fn is not None:
            metric = self.metric_fn(preds, y, num_classes = dataset.base.class_num)
        else:
            metric = (preds == y).float().mean()

        loss = torch.tensor(loss.item(), device=self.device)
        metric = torch.tensor(metric.item(), device=self.device)

        loss = self._reduce_mean(loss)
        metric = self._reduce_mean(metric)

        return loss.item(), metric.item()

    # =========================
    # EMBEDDING (TRAINABLE)
    # =========================
    def _embed_chunked(self, raw_inputs, chunk_size=512):

        chunks = []
        n = next(iter(raw_inputs.values())).shape[0]

        for i in range(0, n, chunk_size):

            chunk = {
                k: v[i:i + chunk_size].to(self.device, non_blocking=True)
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
            labeled_y,
            unlabeled_y
        ], dim=0)

        train_size = labeled_y.size(0)

        train_idx = torch.arange(train_size)
        val_idx = torch.arange(train_size, merged_y.size(0))

        return merged_inputs, merged_y, train_idx, val_idx