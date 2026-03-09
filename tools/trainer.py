import torch
from torch import nn
from torch.utils.data import DataLoader
from tools.metrics import ClassificationMetrics, Metrics
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class TrainerState:
    epoch: int = 0
    batch: int = 0

    train_loss: float | None = None
    val_loss: float | None = None

    train_metrics: Metrics | None = None
    val_metrics: Metrics | None = None

class ModelTrainer:

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion,
            device,
            callbacks=None,
            scheduler=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        self.state = TrainerState()

        self.callbacks = callbacks or []


    def _call_event(self, event):
        """Вызывает методы у коллбэков, если он существует"""

        for cb in self.callbacks:
            func = getattr(cb, event, None)

            if func:
                func(self)
    

    def train(self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        
        self._call_event("on_train_start")

        for epoch in range(num_epochs):
            
            self._call_event("on_epoch_start")

            self.state.epoch = epoch

            self.state.train_loss, self.state.train_metrics = self.train_one_epoch(train_loader=train_loader)
            self.state.val_loss , self.state.val_metrics = self.evalute(val_loader=val_loader)

            if self.scheduler is not None:
                self.scheduler.step()
                
            self._print_progress(num_epochs)

            self._call_event("on_epoch_end")

        self._call_event("on_train_end")

    def _print_progress(self, num_epoch):
            s = self.state
            print(
                f"Эпоха {s.epoch+1} из {num_epoch} -> "
                f"train: loss {s.train_loss:.4f}, acc {s.train_metrics.accuracy:.4f} -> "
                f"val: loss {s.val_loss:.4f}, acc {s.val_metrics.accuracy:.4f}"
            )
    

    def predict(self, dataloader: DataLoader):

        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for img, _ in dataloader:
                img = img.to(self.device)
                logit = self.model(img)
                preds = torch.argmax(logit, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return all_preds

    
    def train_one_epoch(self, train_loader: DataLoader):

        self.model.train()

        metrics = ClassificationMetrics()

        total_loss, total = 0.0, 0.0

        for batch_idx, (img, labels) in enumerate(train_loader):
            
            self._call_event("on_batch_start")
            
            self.state.batch = batch_idx

            img = self._set_device(img)
            labels = self._set_device(labels)

            self.optimizer.zero_grad()

            logits, loss = self._logits_loss(img, labels)

            self._call_event("on_forward_end")

            loss.backward()

            self._call_event("on_backward_end")

            self.optimizer.step()

            total_loss += loss.item() * img.size(0)
            total += labels.size(0)

            metrics.update(logits=logits, labels=labels)

            self._call_event("on_batch_end")

        return total_loss / total, metrics.compute()


    def evalute(self, val_loader: DataLoader):

        self.model.eval()
        metrics = ClassificationMetrics()

        total_loss, total = 0.0, 0.0

        with torch.no_grad():

            for img, labels in val_loader:

                img = self._set_device(img)
                labels = self._set_device(labels)

                logits, loss = self._logits_loss(img, labels)

                total_loss += loss.item() * img.size(0)
                total += labels.size(0)

                metrics.update(logits=logits, labels=labels)


        return total_loss / total, metrics.compute()
    

    def _set_device(self, tensor: torch.Tensor):
        return tensor.to(self.device)

    def _logits_loss(self, img, labels):
        logits = self.model(img)
        return logits, self.criterion(logits, labels)
