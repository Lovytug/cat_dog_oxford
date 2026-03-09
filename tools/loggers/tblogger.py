from tools.loggers.base_callback import Callback
from torch.utils.tensorboard import SummaryWriter


class TBLogger:

    def __init__(self, callbackers: list[Callback], log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self._loggers = [logger(self.writer) for logger in callbackers]

    @property
    def loggers(self):
        return self._loggers
    
    def log_hparams(self, hparams: dict, metrics):
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    def close(self):
        self.writer.close()