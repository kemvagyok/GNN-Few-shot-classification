from models import initalizeModels
from utils.dataset_utils import build_indices

#----------------------------------------------------

from training import get_trainer

#----------------------------------------------------

from preprocessing.indexedDataset import IndexedDataset
from utils import (
    wandb_run,
    save_results,
    is_main_process
)

#----------------------------------------------------

class ExperimentPipeline:
    def __init__(self, config, device, is_ddp, local_rank):
        self.config = config
        self.device = device
        self.is_ddp = is_ddp
        self.local_rank = local_rank

        self.meta = None
        self.datasets = {}
        self.criterion = None
        self.metrics = None

    def set_data(self, train, val, test, meta):
        self.datasets = {
            "train": train,
            "val": val,
            "test": test
        }
        self.meta = meta

    def set_loss(self, criterion):
        self.criterion = criterion

    def set_metrics(self, metrics):
        self.metrics = metrics

    def build_model(self):
        embedder, gnn = initalizeModels(
            config=self.config,
            channel_size=1,
            num_class=self.meta["class_num"],
            latens_size=self.config.latens_size if self.config.train_mode == "full" else self.meta["class_num"],
            device=self.device,
            is_ddp=self.is_ddp,
            input_size=self.meta["input_size"]
        )
        return embedder, gnn

    def run(self, run_id):
        results = []

        original_indices = self.datasets["train"].indices

        for max_label in self.config.train_images_per_class:
            train_idx = self._build_indices(original_indices, max_label)

            train_dataset = IndexedDataset(self.datasets["train"].base, train_idx)

            K_hops = self._resolve_k_hops()

            for K_hop in K_hops:
                with wandb_run(self.config, self.is_ddp, run_id, K_hop, max_label):
                    
                    embedder, gnn = self.build_model()

                    trainer = get_trainer(
                        self.is_ddp,
                        embedder,
                        gnn,
                        self.criterion,
                        self.metrics,
                        self.config,
                        self.device,
                        self.local_rank
                    )

                    metric = trainer.train(
                        train_dataset=train_dataset,
                        val_dataset=self.datasets["val"],
                        test_dataset=self.datasets["test"],
                        K_hop=K_hop
                    )

                results.append((K_hop, max_label, metric))
                
                if is_main_process():
                    save_results(results, self.config, run_id)

        return results

    def _build_indices(self, original_indices, max_label):
        labels = self.meta["labels"][original_indices]

        if max_label == -1:
            return build_indices(labels, max_per_class=len(original_indices))
        return build_indices(labels, max_per_class=max_label)

    def _resolve_k_hops(self):
        if self.config.train_mode == "embedding_only":
            return [None]
        return self.config.K_hop_list