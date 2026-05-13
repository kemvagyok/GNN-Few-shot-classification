from preprocessing.get_dataset import get_dataset
from preprocessing.get_datasetInMemory import get_datasetInMemory

from utils.transformsImage import build_transform
from utils import setup_device
from utils.loss_factory import build_loss


#----------------------------------------------------

from experimentPipeline import ExperimentPipeline

#----------------------------------------------------

from utils.metric_factory import build_metrics
from utils import (
    setup_device
)

class PipelineBuilder:
    def __init__(self, config):
        self.config = config
        self.pipeline = None

    def build_device(self):
        device, local_rank, is_ddp = setup_device()
        self.pipeline = ExperimentPipeline(self.config, device, is_ddp, local_rank)
        return self

    def build_data(self):
        if self.config.fromType == "memory":
            train, val, test, meta = get_datasetInMemory(
                dataset_name=self.config.dataset_name,
                data_pth=self.config.dataset_path,
                img_size=self.config.img_size,
                seed = self.config.seed,
                train_size=self.config.train_size,
                val_size=self.config.val_size,
                test_size=self.config.test_size,
                transform=build_transform(
                    img_size=self.config.img_size, 
                    grayscale=True)
            )
        else:
            train, val, test, meta = get_dataset(
                data_pth=self.config.dataset_path,
                dataset_name=self.config.dataset_name,
                img_dir="images",
                transform=build_transform(
                    img_size=self.config.img_size, 
                    grayscale=True),
                train_size=self.config.train_size,
                val_size=self.config.val_size,
                test_size=self.config.test_size
            )

        self.pipeline.set_data(train, val, test, meta)
        return self

    def build_loss(self):
        criterion = build_loss(
            config=self.config,
            targets=self.pipeline.meta["labels"],
            num_classes=self.pipeline.meta["class_num"],
            device=self.pipeline.device
        )
        self.pipeline.set_loss(criterion)
        return self

    def build_metrics(self):
        metrics = build_metrics(self.config)
        self.pipeline.set_metrics(metrics)
        return self

    def get_pipeline(self):
        return self.pipeline