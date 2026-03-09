import torch
from tools.loggers.base_callback import Callback
from tools.trainer import ModelTrainer
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class ActivityStorage:

    mean: float = 0.0
    std: float = 0.0
    positive: float = 0.0
    effective: float = 0.0
    count: int = 0

    def reset(self):
        self.mean = 0.0
        self.std = 0.0
        self.positive = 0.0
        self.effective = 0.0
        self.count = 0


class FilterActivityLogger(Callback):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.storage = defaultdict(ActivityStorage)
        self.channels = {}
        self.hooks = []


    def on_train_start(self, trainer: ModelTrainer):
         
        for name, module in trainer.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):

                self.channels[name] = module.out_channels

                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):

        def hook(module, inp, output):

            with torch.no_grad():

                output = output.detach()

                mean_act = output.abs().mean(dim=(0, 2, 3))
                std_act = output.std(dim=(0, 2, 3))
                positiv_ratio = (output > 0).float().mean(dim=(0, 2, 3))
                
                A = mean_act
                p = A / (A.sum() + 1e-12)
                effective_filters = 1.0 / (p ** 2).sum()

                stats = self.storage[name]

                stats.mean += mean_act.mean().item()
                stats.std += std_act.mean().item()
                stats.positive += positiv_ratio.mean().item()
                stats.effective += effective_filters.item()
                stats.count += 1

        return hook
    

    def on_epoch_start(self, trainer: ModelTrainer):
        """Сбрасываем статистику перед началом новой эпохи"""
        for stats in self.storage.values():
            stats.reset()


    def on_epoch_end(self, trainer: ModelTrainer):

        epoch = trainer.state.epoch

        for name, stats in self.storage.items():

            if stats.count == 0:
                continue

            channels = self.channels[name]

            mean_activity = stats.mean / stats.count
            mean_std = stats.std / stats.count
            mean_positive = stats.positive / stats.count
            mean_effective = stats.effective / stats.count

            self.writer.add_scalar(
                f"Activation/{name}/mean",
                mean_activity,
                epoch
            )

            self.writer.add_scalar(
                f"Activation/{name}/std",
                mean_std,
                epoch
            )

            self.writer.add_scalar(
                f"Activation/{name}/positive_ratio",
                mean_positive,
                epoch
            )

            self.writer.add_scalar(
                f"Activation/{name}/effective_filters",
                mean_effective,
                epoch
            )

            self.writer.add_scalar(
                f"Activation/{name}/effective_ratio",
                mean_effective / channels,
                epoch
            )


    def on_train_end(self, trainer: ModelTrainer):

        for h in self.hooks:
            h.remove()

            