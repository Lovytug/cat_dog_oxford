import torch


class OptimizerFactory:

    @staticmethod
    def build(model, config):

        name = config["name"]

        param_groups = []

        if "param_groups" in config:

            for group in config["param_groups"]:
                
                if not group.get("use", True):
                    continue

                module_name = group["module"]

                module = getattr(model, module_name)

                param_groups.append({
                    "params": module.parameters(),
                    "lr": group["lr"]
                })

        else:

            param_groups.append({
                "params": model.parameters(),
                "lr": config["lr"]
            })

        if name == "adam":

            return torch.optim.Adam(
                param_groups,
                weight_decay=float(config.get("weight_decay", 0.0))
            )

        raise ValueError(f"Unknown optimizer {name}")