import torch
from torch import nn


class BatchDeepBaselineModel(nn.Module):
    """
        Сеть с добавлением слоев нормализации
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
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

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(128, 37)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        logits = self.fc(x)

        return logits
    


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.relu = nn.ReLU()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        if in_channel == out_channel and stride == 1:
            self.short_cut = nn.Identity()
        else:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = self.short_cut(x)
        x = self.block(x)
        x += identity
        return self.relu(x)
    

class ResidualDeepBaselineModel(nn.Module):
    """
        Создание своей остаточной сети
    """

    def __init__(self):
        super().__init__()

        self.preprocess_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block_1 = nn.Sequential(
            ResidualBlock(32, 32),
            nn.MaxPool2d(2)
        )

        self.block_2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2)
        )

        self.block_3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, 37)


    def forward(self, x):
        x = self.preprocess_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits