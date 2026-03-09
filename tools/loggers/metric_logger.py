from tools.loggers.base_callback import Callback
from tools.trainer import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict


class MetricLogger(Callback):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def on_epoch_end(self, trainer: ModelTrainer):

        s = trainer.state

        self.writer.add_scalar("Loss/train", s.train_loss, s.epoch)
        self.writer.add_scalar("Loss/val", s.val_loss, s.epoch)

        self.writer.add_scalar(
            "Gap/loss",
            s.train_loss - s.val_loss,
            s.epoch
        )

        self._write_metrics(s.train_metrics, s.val_metrics, s.epoch)

    def _write_metrics(self, train_metrics, val_metrics, epoch):

        train_dict = asdict(train_metrics)
        val_dict = asdict(val_metrics)

        for key in train_dict.keys():
            
            self.writer.add_scalar(
                f"Metrics/{key}/train",
                train_dict[key],
                epoch
            )

            self.writer.add_scalar(
                f"Metrics/{key}/val",
                val_dict[key],
                epoch
            )

            self.writer.add_scalar(
                f"Gap/{key}",
                train_dict[key] - val_dict[key],
                epoch
            )