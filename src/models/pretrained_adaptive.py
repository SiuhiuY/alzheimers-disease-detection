import torch
import torch.nn as nn
import torchvision.models as models

# Reference:
# https://discuss.pytorch.org/t/transfer-learning-usage-with-different-input-size/20744/21


class InputAdapter(nn.Module):
    """
    Adapter to convert input with arbitrary number of channels to 3 channels.

    Args:
        in_channels (int): Number of input channels.

    Returns:
        nn.Conv2d: Convolutional layer to adapt input channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Linear transformation to convert in_channels to 3 channels
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PretrainedAlexNet(nn.Module):
    """
    AlexNet model with pretrained weights for feature extraction.
    """
    def __init__(self, trial, input_shape=(4, 32, 32), num_classes=1):
        super().__init__()
        C, _, _ = input_shape

        # Load the pretrained AlexNet model
        base_model = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1
        )

        # Create input adapter
        self.adapter = InputAdapter(C)

        # Rebuild the features block
        features = list(base_model.features.children())
        features = features[:-1]
        features.append(nn.AdaptiveAvgPool2d((6, 6)))
        self.backbone_features = nn.Sequential(*features)

        # Freeze the conv layers
        for param in self.backbone_features.parameters():
            param.requires_grad = False

        # Get hyperparameters for the fully connected layer
        hidden_units = trial.suggest_categorical(
            "alexnet_fc_units", [128]
            )
        dropout_rate = trial.suggest_float("alexnet_fc_dropout", 0.17, 0.47)

        # Replace the classifier with a new fully connected layer
        # .classifier[1] is the first nn.linear layer
        in_features = base_model.classifier[1].in_features
        self.backbone_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.backbone_features(x)
        x = torch.flatten(x, 1)
        x = self.backbone_classifier(x)
        return x


class PretrainedVGG16(nn.Module):
    """
    VGG16 model with pretrained weights for feature extraction.
    """
    def __init__(self, trial, input_shape=(4, 32, 32), num_classes=1):
        super().__init__()
        C, _, _ = input_shape

        # Load the pretrained VGG16 model
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Create input adapter
        self.adapter = InputAdapter(C)

        # Rebuild the features block
        features = list(base_model.features.children())
        features = features[:-1]
        features.append(nn.AdaptiveAvgPool2d((7, 7)))
        self.backbone_features = nn.Sequential(*features)

        # Freeze the conv layers
        for param in self.backbone_features.parameters():
            param.requires_grad = False

        # Get hyperparameters for the fully connected layer
        hidden_units = trial.suggest_categorical(
            "vgg_fc_units", [256]
            )
        dropout_rate = trial.suggest_float("vgg_fc_dropout", 0.1297, 0.2029)

        # Replace the classifier with a new fully connected layer
        in_features = base_model.classifier[0].in_features
        self.backbone_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.backbone_features(x)
        x = torch.flatten(x, 1)
        x = self.backbone_classifier(x)
        return x


class PretrainedResNet18(nn.Module):
    """
    ResNet18 model with pretrained weights for feature extraction.
    """
    def __init__(self, trial, input_shape=(4, 32, 32), num_classes=1):
        super().__init__()
        C, _, _ = input_shape

        # Load the pretrained ResNet18 model
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Create input adapter
        self.adapter = InputAdapter(C)

        # Freeze the conv layers
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        # Define hyperparameters for the fully connected layer
        hidden_units = trial.suggest_categorical(
            "resnet_fc_units", [64, 128, 256]
            )
        dropout_rate = trial.suggest_float("resnet_fc_dropout", 0.1, 0.5)

        # Replace the classifier with a new fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.backbone(x)
        return x
