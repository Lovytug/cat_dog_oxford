import torch
from torch import nn
from torch.utils.data import DataLoader
from tools.tblogger import TBLogger
from tools.metrics import ClassificationMetrics
from collections import defaultdict

class ModelTrainer:

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion,
            device,
            logger: TBLogger = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger

        self._last_val_metric = None
        self._last_val_loss = None

    
    @property
    def val_loss(self):
        return self._last_val_loss
    
    @property
    def val_metric(self):
        return self._last_val_metric


    def train(self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        
        for epoch in range(num_epochs):
            train_loss, train_metrics, mean_ratios = self.train_one_epoch(train_loader=train_loader)
            val_loss, val_metrics = self.evalute(val_loader=val_loader)

            if self.logger is not None:
                self.logger.log_metrics(
                    train_loss=train_loss, train_metric=train_metrics,
                    val_loss=val_loss, val_metric=val_metrics,
                    epoch=epoch
                )
                self.logger.log_gradients(self.model, epoch)
                self.logger.log_update_to_weight_ratio(mean_ratios, epoch)
                self.logger.log_weights(self.model, epoch)

            print(
                f"Эпоха {epoch+1} из {num_epochs} -> "
                f"train: loss {train_loss:.4f}, acc {train_metrics.accuracy:.4f} -> "
                f"val: loss {val_loss:.4f}, acc {val_metrics.accuracy:.4f}"
            )

            self._save_last_val_loss_metric(loss=val_loss, metric=val_metrics)

    def _save_last_val_loss_metric(self, loss, metric):
        self._last_val_loss = loss
        self._last_val_metric = metric

    
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

        total_loss = 0.0
        total = 0

        running_ratio = defaultdict(float)
        count = 0

        for img, labels in train_loader:
            img = img.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, loss = self._logits_loss(img, labels)

            loss.backward()
            
            ratios = self._compute_update_ratio()
            for k, v in ratios.items():
                running_ratio[k] += v
            count += 1

            self.optimizer.step()

            total_loss += loss.item() * img.size(0)
            total += labels.size(0)

            metrics.update(logits=logits, labels=labels)

        metric_values = metrics.compute()

        mean_ratios = {k: v / count for k, v in running_ratio.items()}

        return total_loss / total, metric_values, mean_ratios
    
    def _compute_update_ratio(self):
        ratios = {}

        lr = self.optimizer.param_groups[0]["lr"]

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            grad_norm = p.grad.norm(2)
            weight_norm = p.data.norm(2)

            update_norm = lr * grad_norm
            ratio = update_norm / (weight_norm + 1e-8)

            ratios[name] = ratio.item()

        return ratios


    def evalute(self, val_loader: DataLoader):
        self.model.eval()
        metrics = ClassificationMetrics()

        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for img, labels in val_loader:
                img = img.to(self.device)
                labels = labels.to(self.device)

                logits, loss = self._logits_loss(img, labels)

                total_loss += loss.item() * img.size(0)
                total += labels.size(0)

                metrics.update(logits=logits, labels=labels)

        metric_values = metrics.compute()

        return total_loss / total, metric_values
    

    def _logits_loss(self, img, labels):
        logits = self.model(img)
        return logits, self.criterion(logits, labels)
