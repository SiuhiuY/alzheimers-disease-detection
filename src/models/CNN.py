import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """CNN for 2D feature classification."""

    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()

        # Convolutional blocks
        # Input shape from data loader: (batch_size, window_size, 32, 32)
        # = (batch_size, input_channels, 32, 32)
        # conv1 output shape: (batch_size, 32, 32, 32)
        self.conv1 = nn.Conv2d(input_channels, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # max pool 1 output shape: (batch_size, 32, 16, 16)
        # conv2 output shape: (batch_size, 64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        # max pool 2 output shape: (batch_size, 64, 8, 8)
        self.flatten = nn.Flatten()

        # Fully connected layers
        # fc1 input shape: (batch_size, 64*8*8)
        # Chosen dimension of fc1 output: 128
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CNNModel(input_channels=4, num_classes=1)
