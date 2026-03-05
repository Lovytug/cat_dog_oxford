import torch
from pathlib import Path

from tools.dataset_dataloader import CreaterTrainValDataset, CreaterDataloader
from tools.trainer import ModelTrainer
from tools.tblogger import TBLogger
from tools.transformer_builder import TransformBuilder


class Experiment:

    def __init__(
            self,
            images_dir,
            annotations_dir,
            batch_size=32,
            size_img=(224, 224),
            mean=None,
            std=None,
            device=None
    ):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.size_img = size_img

        self.train_transform = None
        self.val_transform = None
        self.train_loader = None
        self.val_loader = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.trainer = None
        self.logger = None

    
    def setup_transforms(self, train_aug=None):
        builder = TransformBuilder(size_img=self.size_img, mean=self.mean, std=self.std)
        self.train_transform = builder.build_train(augmentations=train_aug)
        self.val_transform = builder.build_val()

    
    def setup_data(self):
        ds_creater = CreaterTrainValDataset(
            images_dir=self.images_dir,
            annotaions_dir=self.annotations_dir,
            transformer_train=self.train_transform,
            transformer_val=self.val_transform
        )
        train_ds = ds_creater.train_dataset()
        val_ds = ds_creater.val_dataset()

        self._create_loaders(train_ds=train_ds, val_ds=val_ds)

    def _create_loaders(self, train_ds, val_ds):
        dl_creater = CreaterDataloader()
        self.train_loader = dl_creater.create(train_ds, self.batch_size)
        self.val_loader = dl_creater.create(val_ds, self.batch_size)
        

    def setup_logger(self, log_dir="runs/exp"):
        self.logger = TBLogger(log_dir=log_dir)
        if self.trainer is not None:
            self.trainer.logger = self.logger


    def setup_model(self, model, optimizer, criterion):
        if model is None or optimizer is None or criterion is None:
            raise ValueError(f"Модель, оптимизатор и критерий должны быть инициализированы. \
                             Пришло {model}, {optimizer}, {criterion}")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainer = ModelTrainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            logger=self.logger
        )

    
    def run(self, epochs=5):
        assert self.train_loader is not None, "Dataloader не был инициализирован. Проверьте наличие setup_data"
        assert self.trainer is not None, "Тренер не был инициализрован. Проверьте наличие setup_model"
        
        self.trainer.train(
            num_epochs=epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader
        )

        if self.logger:
            hparams = self.collect_hparams(epochs)

            metrics = {
                "hparam/final_val_acc": self.trainer.val_metric,
                "hparam/final_val_loss": self.trainer.val_loss
            }
            self.logger.log_hparams(hparams, metrics)
            self.logger.close()


    def collect_hparams(self, epochs):

        return {
            "batch_size": self.batch_size,
            "epochs": epochs,
            "img_size": str(self.size_img),
            "optimizer": self.optimizer.__class__.__name__,
            "lr": self.optimizer.param_groups[0]["lr"],
            "model": self.model.__class__.__name__
        }