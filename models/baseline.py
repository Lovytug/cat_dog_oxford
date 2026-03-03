from torch import nn

class BaselineModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d((2, 2))
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 37)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        return logits