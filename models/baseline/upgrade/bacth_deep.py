import torch
from torch import nn
from models.registry import ModelRegistry


@ModelRegistry.register("batch_new_filter_size_16_baseline")
class BatchDeepBaselineModel(nn.Module):
    """
        Новая модель сети с уменьшением фильтров в начале и расширение горловины перед fc
        углубление классификатора. Добавление dropout
        Структур 3 - 16 - 16 - 32 - 32 - 128 - 128
    """

    def __init__(self, num_classes=37):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        logits = self.classifier(x)

        return logits