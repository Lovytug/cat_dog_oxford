from tools.loggers.base_callback import Callback
from tools.trainer import ModelTrainer
from torch.utils.tensorboard import SummaryWriter

class LRLogger(Callback):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def on_epoch_end(self, trainer: ModelTrainer):

        epoch = trainer.state.epoch

        for i, group in enumerate(trainer.optimizer.param_groups):

            lr = group["lr"]
            name = group.get("name", f"group_{i}")

            self.writer.add_scalar(
                f"LR/{name}",
                lr,
                epoch
            )