import torch


def build_scheduler(optimizer, scheduler_config):

    if scheduler_config is None:
        return None
    
    if not scheduler_config.get("run", False):
        return None

    name = scheduler_config["name"]

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["T_max"]
        )

    if name == "cosine_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config["T_0"]
        )

    raise ValueError(f"Unknown scheduler: {name}")