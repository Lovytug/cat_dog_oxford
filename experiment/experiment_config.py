from dataclasses import dataclass
from tools.metrics import Metrics

@dataclass
class ExperimentConfig:

    experiment_name: str

    model_name: str

    batch_size: int
    epochs: int

    lr: float

    img_size: tuple
    
    optimizer: str


@dataclass
class ExperimentResult:

    experiment_name: str
    model_name: str

    val_metrics: Metrics
    val_loss: float

    epochs: int
    lr: float
    batch_size: int
