import torch
import faiss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import wandb


class TrainerInMemory_DDP:
    def __init__(self, embedder, gnn, criterion, config, local_rank, metric_fn=None):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.config = config

        # Modellek
        self.embedder = DDP(embedder.to(self.device), device_ids=[local_rank], find_unused_parameters=True)
        self.gnn = DDP(gnn.to(self.device), device_ids=[local_rank])

        self.criterion = criterion.to(self.device)
        self.metric_fn = metric_fn

        self.opt_embedder = torch.optim.Adam(self.embedder.parameters(), lr=config.lr_embedder)
        self.opt_gnn = torch.optim.Adam(self.gnn.parameters(), lr=config.lr_gcn)

    # ---------------------------
    # TRAIN
    # ---------------------------
    def train(self, train_dataset, val_dataset=None, test_dataset=None, K_hop=5):
        best_val = 0

        train_sampler = DistributedSampler(
            train_dataset,
            drop_last = False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            drop_last=False
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                sampler=DistributedSampler(val_dataset, shuffle=False),
                drop_last=False
            )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                sampler=DistributedSampler(test_dataset, shuffle=False),
                drop_last=False
            )

        for epoch in range(self.config.epochs_max):
            train_sampler.set_epoch(epoch)

            train_loss = self._train_epoch(train_loader, val_loader, K_hop, epoch)

            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader)

                if dist.get_rank() == 0:
                    print(f"[Epoch {epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | Metric: {val_metric:.4f}")
                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_metric": val_metric
                    }, step=epoch)

                best_val = max(best_val, val_metric)

        if test_loader is not None:
            test_loss, test_metric = self.evaluate(test_loader)
            if dist.get_rank() == 0:
                print(f"Test Loss: {test_loss:.4f} | Metric: {test_metric:.4f}")
                wandb.log({"test_loss" : test_loss, "test_metric": test_metric})
        return best_val

    # ---------------------------
    # TRAIN EPOCH (FIXED)
    # ---------------------------
    def _train_epoch(self, train_loader, val_loader, K_hop, epoch):
        self.embedder.train()
        self.gnn.train()

        local_embeddings = []
        local_labels = []
        local_is_train = []

        # ---------------------------
        # EMBEDDINGS
        # ---------------------------
        with torch.no_grad():  # ⚠️ remove if you want to train embedder
            for batch in train_loader:
                x = self._to_device(batch["inputs"])
                emb = self.embedder(**x)

                local_embeddings.append(emb)
                local_labels.append(batch["labels"].to(self.device))
                local_is_train.append(torch.ones(emb.size(0), dtype=torch.bool, device=self.device))

            if val_loader is not None:
                for batch in val_loader:
                    x = self._to_device(batch["inputs"])
                    emb = self.embedder(**x)

                    local_embeddings.append(emb)
                    local_labels.append(batch["labels"].to(self.device))
                    local_is_train.append(torch.zeros(emb.size(0), dtype=torch.bool, device=self.device))

        local_embeddings = torch.cat(local_embeddings)
        local_labels = torch.cat(local_labels)
        local_is_train = torch.cat(local_is_train)

        # ---------------------------
        # ALL GATHER
        # ---------------------------
        world_size = dist.get_world_size()

        def gather(t):
            out = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(out, t)
            return torch.cat(out)

        global_embeddings = gather(local_embeddings)
        global_labels = gather(local_labels)
        global_train_mask = gather(local_is_train)

        # ---------------------------
        # GRAPH
        # ---------------------------
        graph = self._build_graph(global_embeddings, global_labels, self.config.K_neigh)
        graph.train_mask = global_train_mask

        # ---------------------------
        # SAFE SPLIT
        # ---------------------------
        train_indices = torch.nonzero(global_train_mask).view(-1)

        if train_indices.numel() == 0:
            raise RuntimeError("No training nodes!")

        chunks = train_indices.chunk(world_size)

        if len(chunks) < world_size:
            raise RuntimeError("Too few samples for DDP")

        my_indices = chunks[dist.get_rank()]

        if my_indices.numel() == 0:
            raise RuntimeError(f"Rank {dist.get_rank()} got 0 samples")

        my_mask = torch.zeros_like(global_train_mask)
        my_mask[my_indices] = True

        print(f"[Rank {dist.get_rank()}] nodes: {my_indices.numel()}")

        # ---------------------------
        # LOADER
        # ---------------------------
        loader = NeighborLoader(
            graph,
            num_neighbors=[self.config.K_neigh] * K_hop,
            input_nodes=my_mask,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # ---------------------------
        # TRAIN LOOP
        # ---------------------------
        total_loss = 0
        steps = 0

        for subgraph in loader:
            self.opt_embedder.zero_grad()
            self.opt_gnn.zero_grad()

            subgraph = subgraph.to(self.device)
            out = self.gnn(subgraph)

            mask = my_mask[subgraph.n_id]

            if mask.sum() == 0:
                continue

            loss = self.criterion(out[mask], subgraph.y[mask])
            loss.backward()

            self.opt_embedder.step()
            self.opt_gnn.step()

            total_loss += loss.item()
            steps += 1

        if steps == 0:
            raise RuntimeError(f"Rank {dist.get_rank()} had 0 steps")

        loss = torch.tensor(total_loss / steps, device=self.device)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        return (loss / world_size).item()

    # ---------------------------
    # GRAPH
    # ---------------------------
    def _build_graph(self, embeddings, labels, K):
        emb = embeddings.detach().cpu().numpy()

        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)

        _, I = index.search(emb, K + 1)

        neighbors = torch.tensor(I[:, 1:], device=self.device)

        src = torch.arange(neighbors.size(0), device=self.device).repeat_interleave(K)
        dst = neighbors.flatten()

        edge_index = torch.stack([src, dst], dim=0)

        return Data(x=embeddings, edge_index=edge_index, y=labels)

    # ---------------------------
    # EVAL
    # ---------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        self.embedder.eval()
        self.gnn.eval()

        embs, labels = [], []

        for batch in loader:
            x = self._to_device(batch["inputs"])
            embs.append(self.embedder(**x))
            labels.append(batch["labels"].to(self.device))

        embs = torch.cat(embs)
        labels = torch.cat(labels)

        world_size = dist.get_world_size()

        def gather(t):
            out = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(out, t)
            return torch.cat(out)

        embs = gather(embs)
        labels = gather(labels)

        graph = self._build_graph(embs, labels, self.config.K_neigh)
        out = self.gnn(graph)

        loss = self.criterion(out, labels)
        preds = out.argmax(dim=1)

        metric = self.metric_fn(preds, labels, loader.dataset.base.class_num) \
            if self.metric_fn else (preds == labels).float().mean().item()

        return loss.item(), metric

    def _to_device(self, x):
        return {k: v.to(self.device) for k, v in x.items()}