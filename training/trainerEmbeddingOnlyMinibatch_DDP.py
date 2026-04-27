import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import EarlyStopping
import wandb


class TrainerEmbeddingOnlyMinibatch_DDP:
    def __init__(self, embedder, criterion, config, rank, world_size, metric_fn=None):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        self.embedder = embedder.to(self.device)
        self.embedder = DDP(self.embedder, device_ids=[rank])

        self.criterion = criterion
        self.metric_fn = metric_fn
        self.config = config

        self.optimizer = torch.optim.Adam(
            self.embedder.parameters(),
            lr=config.lr_embedder
        )

    def _create_loader(self, dataset, shuffle=True):
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            pin_memory=True
        ), sampler

    def train(self, train_dataset, val_dataset=None, test_dataset=None):
        train_loader, train_sampler = self._create_loader(train_dataset, shuffle=True)

        if val_dataset is not None:
            val_loader, _ = self._create_loader(val_dataset, shuffle=False)

        best_val_metric = 0
        stopper = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.delta
        )
        for epoch in range(self.config.epochs_max):
            train_sampler.set_epoch(epoch)

            train_loss = self._train_epoch(train_loader)

            if val_dataset is not None:
                val_loss, val_metric = self.evaluate(val_loader)

                if self.rank == 0:
                    print(f"[Epoch {epoch}] "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Metric: {val_metric:.4f}")

                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_metric": val_metric
                    }, step=epoch)

                if val_metric > best_val_metric:
                    best_val_metric = val_metric

            if stopper.early_stop(val_loss):
                if self.rank == 0:
                    print("Early stopping triggered.")
                    wandb.log({"early_stopping_epoch": epoch})
                break


        if self.rank == 0:
            torch.save(
                self.embedder.module.state_dict(),
                f"{self.config.model_embedder_save_path}/model.pth"
            )

        if test_dataset is not None:
            test_loader, _ = self._create_loader(test_dataset, shuffle=False)
            test_loss, test_metric = self.evaluate(test_loader)

            if self.rank == 0:
                print(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f}")
                wandb.log({
                    "test_loss": test_loss,
                    "test_metric": test_metric
                })

        return best_val_metric

    def _train_epoch(self, loader):
        self.embedder.train()

        total_loss = 0
        num_batches = 0

        for batch in loader:
            inputs = {
                k: v.to(self.device)
                for k, v in batch["inputs"].items()
            }
            y = batch["labels"].to(self.device)

            loss = self._step(inputs, y)

            total_loss += loss.item()
            num_batches += 1

        loss_tensor = torch.tensor(total_loss / max(num_batches, 1)).to(self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= self.world_size

        return loss_tensor.item()

    def _step(self, inputs, y):
        self.optimizer.zero_grad()

        embeddings = self.embedder(**inputs)
        loss = self.criterion(embeddings, y)

        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def evaluate(self, loader):
        self.embedder.eval()

        total_loss = 0
        num_batches = 0

        all_preds = []
        all_y = []

        for batch in loader:
            inputs = {
                k: v.to(self.device)
                for k, v in batch["inputs"].items()
            }
            y = batch["labels"].to(self.device)

            embeddings = self.embedder(**inputs)
            loss = self.criterion(embeddings, y)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(embeddings.detach())
            all_y.append(y.detach())

        local_preds = torch.cat(all_preds)
        local_y = torch.cat(all_y)

        # --- GATHER ---
        gathered_preds = [torch.zeros_like(local_preds) for _ in range(self.world_size)]
        gathered_y = [torch.zeros_like(local_y) for _ in range(self.world_size)]

        dist.all_gather(gathered_preds, local_preds)
        dist.all_gather(gathered_y, local_y)

        preds = torch.cat(gathered_preds)
        y_all = torch.cat(gathered_y)

        loss_tensor = torch.tensor(total_loss / max(num_batches, 1)).to(self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= self.world_size

        if self.metric_fn is not None:
            metric_value = self.metric_fn(preds.cpu(), y_all.cpu())
        else:
            metric_value = (preds.argmax(dim=1) == y_all).float().mean().item()

        return loss_tensor.item(), metric_value