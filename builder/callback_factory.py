from tools.loggers.metric_logger import MetricLogger
from tools.loggers.gradients_logger import GradientsLogger
from tools.loggers.weight_update_logger import WeightUpdateLogger
from tools.loggers.filter_activ_logger import FilterActivityLogger
from tools.loggers.lr_logger import LRLogger


class CallbackFactory:

    registry = {

        "metric_logger": MetricLogger,
        "gradients_logger": GradientsLogger,
        "weight_update_logger": WeightUpdateLogger,
        "filter_activity_logger": FilterActivityLogger,
        "lr_logger": LRLogger,
    }


    @classmethod
    def build(cls, config):

        callbacks = []

        for name in config:

            if name not in cls.registry:
                raise ValueError(f"Неизвестный коллбекер {name}")

            callbacks.append(cls.registry[name])

        return callbacks