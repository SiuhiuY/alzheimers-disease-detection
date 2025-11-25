import torch
import os
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
from src.feature_loader import load_features

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


def min_max_normalise(features, train_ind, test_ind):
    """Apply Min-Max normalisation to features based on training data.

    Args:
        features (np.ndarray): The feature data to normalise.
        train_ind (list or np.ndarray): Indices of training samples.
        test_ind (list or np.ndarray): Indices of testing samples.

    Raises:
        ValueError: If normalised training features are not in [0, 1].

    Returns:
        tuple: Normalised (train_features, test_features).
    """
    train_features = features[train_ind]
    test_features = features[test_ind]

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


if __name__ == "__main__":
    AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
    AD_CN = {"A": 0, "C": 1}
    FTD_CN = {"F": 0, "C": 1}
    features, labels, subjects = load_features(
        label_map=FTD_CN, band="alpha"
    )
    class_weights = calculate_class_weights(labels)
    print("Class weights:", class_weights)
