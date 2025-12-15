import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

# Reference:
# https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py


class OptunaCNN(nn.Module):
    """
    CNN model for 2D feature classification with Optuna hyperparameter
    tuning.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter
            suggestions.
        input_shape (tuple): Shape of the input data (channels, height,
            width).
        num_classes (int): Number of output classes.
    """
    def __init__(
        self,
        trial: optuna.Trial,
        input_shape=(4, 32, 32),
        num_classes=1
    ):
        super().__init__()
        C, H, W = input_shape
        self.input_channels = C
        self.height = H
        self.width = W
        self.num_classes = num_classes

        # Architecture hyperparameters
        # Number of convolutional layers
        n_layers = trial.suggest_categorical("n_layers", [2, 3])
        layers = []

        for i in range(n_layers):
            # Output channels per conv layer
            conv_out_channels = trial.suggest_categorical(
                f"filters_{i}", [32, 64, 96]
                )

            # Kernel size per conv layer
            kernel_size = 3

            # BatchNorm per conv layer
            # padding=kernel_size // 2 to maintain spatial size
            if i == 0:
                use_batchnorm = True
            else:
                use_batchnorm = trial.suggest_categorical(
                    f"conv_{i}_use_batchnorm", [True, False])
            layers.append(nn.Conv2d(
                self.input_channels, conv_out_channels,
                kernel_size, padding=kernel_size // 2))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(conv_out_channels))

            # Activation function per conv layer
            activation_name = trial.suggest_categorical(
                f"conv_{i}_activation", ["relu", "leaky_relu"])
            activation = nn.ReLU() if activation_name == "relu" \
                else nn.LeakyReLU()
            layers.append(activation)

            # Pooling per conv layer
            pooling_type = "max"
            if self.height > 2 and self.width > 2:
                layers.append(
                    nn.MaxPool2d(2) if pooling_type == "max"
                    else nn.AvgPool2d(2)
                )
                self.height //= 2
                self.width //= 2

            # Update input channels for next layer
            self.input_channels = conv_out_channels

        # Flatten layer
        layers.append(nn.Flatten())
        flatten_dim = self.input_channels * self.height * self.width

        # Fully connected layer hyperparameters
        input_features = flatten_dim
        n_fc_layers = trial.suggest_categorical(
            "n_fc_layers", [2, 3]
            )
        for i in range(n_fc_layers):
            out_features = trial.suggest_categorical(
                f"fc_{i}_out_features", [128, 192, 256]
                )
            layers.append(nn.Linear(input_features, out_features))

            # BatchNorm per fc layer
            fc_batchnorm = trial.suggest_categorical(
                f"fc_{i}_use_batchnorm", [True, False]
            )
            if fc_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))

            # Activation function per fc layer
            fc_activation = trial.suggest_categorical(
                f"fc_{i}_activation", ["relu", "leaky_relu"]
            )
            activation = nn.ReLU() if fc_activation == "relu" \
                else nn.LeakyReLU()
            layers.append(activation)

            # Dropout per fc layer
            dropout_rate = trial.suggest_float(
                f"fc_{i}_dropout_rate", 0.05, 0.35
            )
            layers.append(nn.Dropout(dropout_rate))

            # Update input features for next layer
            input_features = out_features

        # Output layer
        layers.append(nn.Linear(input_features, self.num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
