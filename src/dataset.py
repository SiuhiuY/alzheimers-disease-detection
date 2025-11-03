from torch.utils.data import Dataset
import torch


class EEGDataset(Dataset):
    """Custom Dataset for EEG data."""

    def __init__(self, features, labels, subjects):
        """Initialize the EEGDataset with data and labels.

        Args:
            features (np.ndarray): Feature data.
            labels (np.ndarray): Corresponding class labels.
            subjects (np.ndarray): Corresponding subject IDs.
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.subjects = subjects

    def __len__(self):
        """Return the total number of windows in the dataset."""
        return self.features.shape[0]

    def __getitem__(self, idx):
        """Retrieve the features, labels, and subject IDs for a given index."""
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        subjects = self.subjects[idx]
        return features, labels, subjects
