import torch
import os
import numpy as np
import random

# Reference:
# https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031


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
    os.environ['PYTHONHASHSEED'] = str(seed)


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
