from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


@dataclass
class Metrics:

    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


class ClassificationMetrics:

    def __init__(self):
        self.reset()

    def reset(self):
        self.targets = []
        self.preds = []

    def update(self, logits, labels):
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        self.preds.extend(preds)
        self.targets.extend(labels)

    def compute(self) -> Metrics:
        return Metrics(
            accuracy=accuracy_score(self.targets, self.preds),
            balanced_accuracy=balanced_accuracy_score(self.targets, self.preds),
            precision_macro=precision_score(self.targets, self.preds, average="macro", zero_division=0),
            recall_macro=recall_score(self.targets, self.preds, average="macro", zero_division=0),
            f1_macro=f1_score(self.targets, self.preds, average="macro", zero_division=0)
        )