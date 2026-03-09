from tools.loggers.base_callback import Callback
from tools.trainer import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class GradientsLogger(Callback):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.layer_norms = defaultdict(list)


    def on_epoch_start(self, trainer: ModelTrainer):
        self.layer_norms.clear()


    def on_backward_end(self, trainer: ModelTrainer):

        for name, param in trainer.model.named_parameters():

            if param.grad is None:
                continue

            grad = param.grad.detach()

            norm = grad.norm(2).item()

            self.layer_norms[name].append(norm)


    def on_epoch_end(self, trainer: ModelTrainer):

        epoch = trainer.state.epoch

        total_norm_sq = 0.0
        for name, norms in self.layer_norms.items():
                
                mean_norm = sum(norms) / len(norms)

                total_norm_sq += mean_norm ** 2

                self.writer.add_scalar(
                    f"GradNorm/layer/{name}",
                    mean_norm,
                    epoch
                )

        global_norm = total_norm_sq ** 0.5

        self.writer.add_scalar(
            "GradNorm/global",
            global_norm,
            epoch
        )