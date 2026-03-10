import torch


class SchedulerFactory:

    @staticmethod
    def build(optimizer, config):

        if config is None:
            return None

        if not config.get("run", True):
            return None

        name = config["name"]

        if name == "cosine":

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["T_max"]
            )

        if name == "cosine_restart":

            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config["T_0"]
            )

        raise ValueError(name)