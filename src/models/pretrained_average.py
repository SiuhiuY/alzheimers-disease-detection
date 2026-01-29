import torch
import torch.nn as nn
import torchvision.models as models

# Reference:
# https://github.com/bnsreenu/python_for_microscopists/blob/master/Tips_tricks_20_Understanding%20transfer%20learning%20for%20different%20size%20and%20channel%20inputs.py
# https://deeplizard.com/learn/video/stWU37L91Yc
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://docs.pytorch.org/vision/main/_modules/torchvision/models/alexnet.html#alexnet
# https://docs.pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg16
# https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18


def expand_first_conv(conv, input_channels):
    """
    Expand the first convolutional layer by using the average of the existing
    weights.

    Args:
        conv (nn.Conv2d): Original convolutional layer.
        input_channels (int): Intended number of input channels.

    Returns:
        nn.Conv2d: New convolutional layer with expanded input channels.
    """
    out_channels = conv.out_channels
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    bias = conv.bias is not None  # bias=False in ResNet18 first conv

    # Initialise new convolutional layer
    new_conv = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)

    # Copy weights from the original conv layer
    # Prevent gradient tracking through the weight-copying operation
    with torch.no_grad():
        # Copy existing weights for the first 3 channels
        new_conv.weight[:, :3] = conv.weight
        # Use the mean of the existing weights for the additional channels
        if input_channels > 3:
            # weight shape: (out_channels, 3, kernel_h, kernel_w)
            # mean_weights shape: (out_channels, 1, kernel_h, kernel_w)
            mean_weights = conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, input_channels):
                new_conv.weight[:, i:i+1] = mean_weights
        if bias:
            new_conv.bias = conv.bias

    return new_conv


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
        # Modify the first convolutional layer to accept additional channels
        base_model.features[0] = expand_first_conv(base_model.features[0], C)

        # Iterate through the conv and pooling layers
        features = list(base_model.features.children())
        # Exclude the last MaxPool2d to avoid too much downsampling
        # when input size is small (32x32)
        features = features[:-1]
        # AdaptiveAvgPool2d in the original architecture
        features.append(nn.AdaptiveAvgPool2d((6, 6)))
        # Wrap the modified layers in a Sequential module
        self.backbone_features = nn.Sequential(*features)

        # Freeze the convolutional layers
        for param in self.backbone_features.parameters():
            param.requires_grad = False

        # Define hyperparameters in the fully connected layer
        hidden_units = trial.suggest_categorical(
            "alexnet_fc_units", [64, 128, 256]
        )
        dropout_rate = trial.suggest_float(
            "alexnet_fc_dropout", 0.1, 0.5
        )

        # Replace the classifier with a new fully connected layer
        # Get the number of input features to the classifier
        # (256 * 6 * 6 = 9216)
        # .classifier[0] is nn.Dropout
        # .classifier[1] is the first nn.linear layer
        in_features = base_model.classifier[1].in_features
        self.backbone_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
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

        # Modify the first convolutional layer to accept additional channels
        base_model.features[0] = expand_first_conv(base_model.features[0], C)
        features = list(base_model.features.children())
        features = features[:-1]
        features.append(nn.AdaptiveAvgPool2d((7, 7)))
        self.backbone_features = nn.Sequential(*features)

        # Freeze the convolutional layers
        for param in self.backbone_features.parameters():
            param.requires_grad = False

        # Define hyperparameters in the fully connected layer
        hidden_units = trial.suggest_categorical(
            "vgg_fc_units", [64, 128, 256]
        )
        dropout_rate = trial.suggest_float(
            "vgg_fc_dropout", 0.1, 0.5
        )

        # Replace the classifier with a new fully connected layer
        # Get the number of input features to the classifier
        # (512 * 7 * 7 = 25088)
        in_features = base_model.classifier[0].in_features
        self.backbone_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
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
        # Modify the first convolutional layer to accept additional channels
        self.backbone.conv1 = expand_first_conv(self.backbone.conv1, C)

        # Freeze all layers except the final fully connected layer
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        # Define hyperparameters in the fully connected layer
        hidden_units = trial.suggest_categorical(
            "resnet_fc_units", [64, 128, 256]
        )
        dropout_rate = trial.suggest_float(
            "resnet_fc_dropout", 0.1, 0.5
        )

        # Replace the fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
