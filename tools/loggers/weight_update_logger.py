from tools.loggers.base_callback import Callback
from tools.trainer import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class WeightUpdateLogger(Callback):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.layer_ratio = defaultdict(list)
        self.param_lr = {}

    def on_train_start(self, trainer: ModelTrainer):

        for group in trainer.optimizer.param_groups:

            lr = group["lr"]

            for param in group["params"]:
                self.param_lr[id(param)] = lr


    def on_epoch_start(self, trainer: ModelTrainer):
        self.layer_ratio.clear()


    def on_backward_end(self, trainer: ModelTrainer):

        for name, param in trainer.model.named_parameters():

            if param.grad is None:
                continue

            weight = param.data.detach()
            grad = param.grad.detach()

            norm_w = weight.norm(2).item()
            norm_g = grad.norm(2).item()
            
            lr = float(self.param_lr.get(id(param), trainer.optimizer.param_groups[0]["lr"]))

            ratio = lr * norm_g / (norm_w + 1e-10)

            self.layer_ratio[name].append(ratio)


    def on_epoch_end(self, trainer: ModelTrainer):

        epoch = trainer.state.epoch

        total_norm_sq = 0.0
        for name, ratio in self.layer_ratio.items():
                
            mean_norm = sum(ratio) / len(ratio)

            total_norm_sq += mean_norm**2

            self.writer.add_scalar(
                f"WeightUpdateRatio/layer/{name}",
                mean_norm,
                epoch
            )

        global_ratio = total_norm_sq ** 0.5

        self.writer.add_scalar(
            "WeightUpdateRatio/global", 
            global_ratio, 
            epoch
        )

    def get_lr_for_param(self, trainer, param):

        for group in trainer.optimizer.param_groups:
            if param in group["params"]:
                return group["lr"]

        return trainer.optimizer.param_groups[0]["lr"]

            