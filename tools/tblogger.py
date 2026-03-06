import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from dataclasses import asdict

class TBLogger:

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    
    def log_metrics(
            self,
            train_loss,
            train_metric,
            val_loss,
            val_metric,
            epoch
    ):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)

        train_dict = asdict(train_metric)
        val_dict = asdict(val_metric)

        for name, value in train_dict.items():
            self.writer.add_scalar(f"Train/{name}", value, epoch)

        for name, value in val_dict.items():
            self.writer.add_scalar(f"Val/{name}", value, epoch)


    def log_gradients(self, model: nn.Module, epoch: int):
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None: # смотрим только на НЕ замороженные слои
                grad_norm = param.grad.data.norm(2)
                total_norm += grad_norm.item()

                self.writer.add_histogram(
                    f"Gradients/{name}",
                    param.grad,
                    epoch
                )

        self.writer.add_scalar(
            "GradNorm/total",
            total_norm,
            epoch
        )


    def log_weights(self, model: nn.Module, epoch: int):
        for name, param in model.named_parameters():
            self.writer.add_histogram(
                f"Weights/{name}",
                param.data,
                epoch
            )


    def log_hparams(self, hparams: dict, metrics):
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    
    def close(self):
        self.writer.close()