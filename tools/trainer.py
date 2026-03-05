import torch
from torch import nn
from torch.utils.data import DataLoader
from tools.tblogger import TBLogger

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
            train_loss, train_met = self.train_one_epochs(train_loader=train_loader)
            val_loss, val_met = self.evalute(val_loader=val_loader)

            if self.logger is not None:
                self.logger.log_metrics(
                    train_loss=train_loss, train_metric=train_met,
                    val_loss=val_loss, val_metric=val_met,
                    epoch=epoch
                )
                self.logger.log_gradients(self.model, epoch)
                self.logger.log_weights(self.model, epoch)

            print(
                f"Эпоха {epoch+1} из {num_epochs} -> "
                f"train: loss {train_loss:.4f}, acc {train_met:.4f} -> "
                f"val: loss {val_loss:.4f}, acc {val_met:.4f}"
            )

            self._save_last_val_loss_metric(loss=val_loss, metric=val_met)

    def _save_last_val_loss_metric(self, loss, metric):
        self.last_val_loss = loss
        self.last_val_metric = metric

    
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

    
    def train_one_epochs(self, train_loader: DataLoader):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for img, labels in train_loader:
            img = img.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, loss = self._logits_loss(img, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * img.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total
    

    def evalute(self, val_loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for img, labels in val_loader:
                img = img.to(self.device)
                labels = labels.to(self.device)

                logits, loss = self._logits_loss(img, labels)

                total_loss += loss.item() * img.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total
    

    def _logits_loss(self, img, labels):
        logits = self.model(img)
        return logits, self.criterion(logits, labels)
