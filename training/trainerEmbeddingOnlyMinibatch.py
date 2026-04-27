import torch
from torch.utils.data import DataLoader
import wandb

class TrainerEmbeddingOnlyMinibatch:
    def __init__(self, embedder, criterion, config, device, metric_fn=None):
        self.embedder = embedder.to(device)
        self.device = device
        self.config = config
        self.criterion = criterion
        self.metric_fn = metric_fn

        self.optimizer = torch.optim.Adam(
            self.embedder.parameters(),
            lr=config.lr_embedder
        )

    def _create_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            pin_memory=True
        )

    def train(self, train_dataset, val_dataset=None, test_dataset=None, **kwargs):
        train_loader = self._create_loader(train_dataset, shuffle=True)

        if val_dataset is not None:
            val_loader = self._create_loader(val_dataset, shuffle=False)

        best_val_metric = 0

        for epoch in range(self.config.epochs_max):
            train_loss = self._train_epoch(train_loader)

            if val_dataset is not None:
                val_loss, val_metric = self.evaluate(val_loader)

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
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        torch.save(
            self.embedder.state_dict(),
            f"{self.config.model_embedder_save_path}/model.pth"
        )

        if test_dataset is not None:
            test_loader = self._create_loader(test_dataset, shuffle=False)
            test_loss, test_metric = self.evaluate(test_loader)

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

        return total_loss / max(num_batches, 1)

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

        preds = torch.cat(all_preds)
        y_all = torch.cat(all_y)

        if self.metric_fn is not None:
            metric_value = self.metric_fn(preds.cpu(), y_all.cpu())
        else:
            metric_value = (preds.argmax(dim=1) == y_all).float().mean().item()

        return total_loss / max(num_batches, 1), metric_value