import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """CNN for EEG classification."""

    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CNNModel(input_channels=4, num_classes=3)
