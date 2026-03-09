from tools.loggers.base_callback import Callback
from tools.trainer import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class WeightUpdateLogger(Callback):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.layer_ratio = defaultdict(list)


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

            self.layer_ratio[name].append(norm_g / norm_w)


    def on_epoch_end(self, trainer: ModelTrainer):

        epoch = trainer.state.epoch

        lr = trainer.optimizer.param_groups[0]['lr']
        total_norm_sq = 0.0
        for name, ratio in self.layer_ratio.items():
                
            mean_norm = sum(ratio) / len(ratio)
            R = lr * mean_norm

            total_norm_sq += R**2

            self.writer.add_scalar(
                f"WeightUpdateRatio/layer/{name}",
                R,
                epoch
            )

        global_ratio = total_norm_sq ** 0.5

        self.writer.add_scalar(
            "WeightUpdateRatio/global", 
            global_ratio, 
            epoch
        )

            