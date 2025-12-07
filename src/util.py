import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from src.feature_loader import load_features
from src.dataset import EEGDataset

# Reference:
# https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031
# https://discuss.pytorch.org/t/compute-class-weight/28379
# https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch


def reproducability(seed):
    """Set seeds for reproducibility across different libraries.

    Args:
        seed (int): The seed value to set.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def min_max_normalise(features, train_idx, test_idx):
    """Apply Min-Max normalisation to features based on training data.

    Args:
        features (np.ndarray): The feature data to normalise.
        train_idx (list or np.ndarray): Indices of training samples.
        test_ind (list or np.ndarray): Indices of testing samples.

    Raises:
        ValueError: If normalised training features are not in [0, 1].

    Returns:
        tuple: Normalised (train_features, test_features).
    """
    train_features = features[train_idx]
    test_features = features[test_idx]

    # Compute min and max for pixels across all windows in training set
    train_min = train_features.min()
    train_max = train_features.max()

    eps = 1e-8  # to avoid division by zero
    train_features = (train_features - train_min) / (
        train_max - train_min + eps)
    test_features = (test_features - train_min) / (
        train_max - train_min + eps)

    # Check if train features are in [0, 1] with a tolerance
    if not (np.isclose(train_features.min(), 0, atol=1e-6) or
            train_features.min() >= 0):
        raise ValueError("Train min value below 0 after normalisation.")
    if not (np.isclose(train_features.max(), 1, atol=1e-6) or
            train_features.max() <= 1):
        raise ValueError("Train max value above 1 after normalisation.")

    return train_features, test_features


def calculate_class_weights(labels):
    """Calculate class weights to handle class imbalance.

    Args:
        labels (np.ndarray): Array of class labels.

    Returns:
        dict: A dictionary mapping class indices to their weights.
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    # Calculate positive class weight for BCEWithLogitsLoss
    positive_weight = weights[1] / weights[0]
    # Convert to tensor
    positive_weight = torch.tensor(positive_weight, dtype=torch.float32)

    return positive_weight


def get_criterion(use_class_weights, labels, device):
    """Get the loss criterion, optionally with class weights.

    Args:
        use_class_weights (bool): Whether to use class weights.
        labels (np.ndarray): Array of class labels.
        device (torch.device): Device to place the weights tensor on.

    Returns:
        torch.nn.Module: The loss criterion.
    """
    if use_class_weights:
        pos_weight = calculate_class_weights(labels).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    return criterion


def get_data_loaders(features, labels, subjects,
                     train_idx, test_idx,
                     train_transform=None, batch_size=64,
                     shuffle=True):
    """Create DataLoaders for training and testing datasets.
    Args:
        features (np.ndarray): Feature data.
        labels (np.ndarray): Corresponding labels.
        subjects (np.ndarray): Subject identifiers.
        train_idx (list or np.ndarray): Indices for training samples.
        test_idx (list or np.ndarray): Indices for testing samples.
        train_transform (callable, optional):
            Transform to apply to training data.
        batch_size (int, optional): Batch size for DataLoaders.
        shuffle (bool, optional): Whether to shuffle training data.
    """
    train_set = EEGDataset(
        features[train_idx],
        labels[train_idx],
        subjects[train_idx],
        transform=train_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_set = EEGDataset(
        features[test_idx],
        labels[test_idx],
        subjects[test_idx],
        transform=None
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_optimizer(optimizer_name, model_parameters,
                  learning_rate, weight_decay=0.0):
    """Get the optimizer based on the given name.

    Args:
        optimizer_name (str): Name of the optimizer.
        model_parameters (iterable): Model parameters to optimize.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The instantiated optimizer.
    """
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer


if __name__ == "__main__":
    AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
    AD_CN = {"A": 0, "C": 1}
    FTD_CN = {"F": 0, "C": 1}
    features, labels, subjects = load_features(
        label_map=FTD_CN, band="alpha"
    )
    class_weights = calculate_class_weights(labels)
    print("Class weights:", class_weights)
