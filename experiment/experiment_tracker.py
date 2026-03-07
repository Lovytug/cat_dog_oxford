import csv
from pathlib import Path
from dataclasses import asdict
from experiment.experiment_config import ExperimentResult
from tools.metrics import Metrics


class ExperimentTracker:

    def __init__(self, file_path="experiments/results.csv"):

        self.file_path = Path(file_path)

        if not self.file_path.exists():
            self._create_file()

    def _create_file(self):

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file_path, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "experiment_name",
                "model_name",
                "epochs",
                "lr",
                "batch_size",

                "val_loss",

                "accuracy",
                "balanced_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro"
            ])

    def save(self, result: ExperimentResult):

        metrics: Metrics = result.val_metrics

        with open(self.file_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                result.experiment_name,
                result.model_name,
                result.epochs,
                result.lr,
                result.batch_size,

                (result.val_loss),

                (metrics.accuracy),
                (metrics.balanced_accuracy),
                (metrics.precision_macro),
                (metrics.recall_macro),
                (metrics.f1_macro)
            ])