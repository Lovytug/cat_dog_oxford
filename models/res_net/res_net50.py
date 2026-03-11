import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from models.registry import ModelRegistry


@ModelRegistry.register("resnet50_transfer")
class ResNetTransfer(nn.Module):

    def __init__(self, num_classes=37, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        for p in self.parameters():
            p.requires_grad = False


    def forward(self, x):

        x = self.backbone(x)

        logits = self.fc(x)

        return logits