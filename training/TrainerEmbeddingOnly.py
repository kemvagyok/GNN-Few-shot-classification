import wandb
import torch

class TrainerEmbeddingOnly:
    def __init__(self, embedder, criterion, config, device, metric_fn=None):
        self.embedder = embedder.to(device)
        self.device = device
        self.config = config
        self.criterion = criterion  # embedding-alapú loss
        self.metric_fn = metric_fn

        self.optimizer = torch.optim.Adam(
            self.embedder.parameters(),
            lr=config.lr_embedder
        )

    def train(self, train_dataset, val_dataset=None):
        best_val_metric = 0

        for epoch in range(self.config.epochs_max):
            train_loss = self._train_epoch(train_dataset)

            if val_dataset is not None:
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

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        return best_val_metric


    def _train_epoch(self, dataset):
        self.embedder.train()

        data = dataset.get_all()
        inputs = data["inputs"]
        y = data["labels"]
        train_mask = data.get("train_mask", None)

        n = y.shape[0]
        indices = torch.arange(n)

        if train_mask is not None:
            indices = indices[train_mask]
            print(f"Training on {len(indices)} samples")

        # 🔀 shuffle
        perm = indices[torch.randperm(indices.size(0))]

        batch_size = self.config.batch_size
        total_loss = 0
        num_batches = 0

        for i in range(0, perm.size(0), batch_size):
            batch_idx = perm[i:i + batch_size]

            batch_inputs = {
                k: v[batch_idx].to(self.device)
                for k, v in inputs.items()
            }
            batch_y = y[batch_idx].to(self.device)

            embeddings = self.embedder(**batch_inputs)

            loss = self.criterion(embeddings, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


    @torch.no_grad()
    def evaluate(self, dataset):
        self.embedder.eval()

        data = dataset.get_all()
        inputs = data["inputs"]
        y = data["labels"].to(self.device)

        n = y.shape[0]
        batch_size = self.config.batch_size

        all_embeddings = []

        for i in range(0, n, batch_size):
            batch_inputs = {
                k: v[i:i + batch_size].to(self.device)
                for k, v in inputs.items()
            }

            emb = self.embedder(**batch_inputs)
            all_embeddings.append(emb)

        embeddings = torch.cat(all_embeddings, dim=0)

        loss = self.criterion(embeddings, y).item()

        if self.metric_fn is not None:
            metric_value = self.metric_fn(embeddings, y)
        else:
            metric_value = (embeddings.argmax(dim=1) == y).sum().item() / y.size(0)

        return loss, metric_value