from torch import nn
from models.registry import ModelRegistry

@ModelRegistry.register("short_baseline_model")
class ShortBaselineModel(nn.Module):
    """
        Неглубокая сеть сверточная сеть из трех сверток.
        Нет вспомогательных слоев
    """

    def __init__(self, num_classes=37):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d((2, 2))
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)


    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))

        x = self.pool(self.relu(self.conv2(x)))

        x = self.relu(self.conv3(x))

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        return logits
    
@ModelRegistry.register("deep_baseline_model")
class DeepBaselineModel(nn.Module):
    """
        Глубокая сверточная сеть из трех блоков, по двум сверточным слоям.
        Нет вспомогательных слоев
    """

    def __init__(self, num_classes=37):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(128, num_classes)


    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        logits = self.fc(x)

        return logits