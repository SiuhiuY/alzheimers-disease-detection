import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import src.util as util
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from src.feature_loader import load_features
from src.dataset import EEGDataset
from src.models.CNN import CNNModel
from src.model_trainer import ModelTrainer

# Reference:
# https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define data configuration
LABEL_PATH = "data/participants.tsv"
AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
AD_CN = {"A": 1, "C": 0}
FTD_CN = {"F": 1, "C": 0}
AD_FTD = {"A": 1, "F": 0}
label_map = FTD_CN
BAND = "alpha"

# Define training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4


def quick_run(model_name, test_size=0.2, seed=123):
    """Run the model training and evaluation pipeline."""

    # Set random seeds
    util.reproducability(seed)

    # Load data
    features, labels, subjects = load_features(
        label_map=label_map, band=BAND
    )

    # Create a single train/test split stratified by class
    # and grouped by subject.
    single_split = StratifiedGroupKFold(
        n_splits=int(1/test_size), shuffle=True, random_state=seed)

    train_ind, test_ind = next(single_split.split(
        features, labels, groups=subjects
    ))

    if len(
        set(np.unique(subjects[train_ind])) &
        set(np.unique(subjects[test_ind]))
    ) != 0:
        raise ValueError("Subjects overlap between train and test sets!")

    # Min-Max normalisation
    train_features, test_features = util.min_max_normalise(
        features, train_ind, test_ind
    )

    train_subset = EEGDataset(
        train_features, labels[train_ind], subjects[train_ind]
    )
    test_subset = EEGDataset(
        test_features, labels[test_ind], subjects[test_ind]
    )

    # Create DataLoaders for training and testing
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Initialise model
    model = model_name.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = ModelTrainer(
        model, train_loader, test_loader, optimizer, criterion, DEVICE
    )

    # Training loop
    trainer.fit(NUM_EPOCHS)
    # for epoch in range(NUM_EPOCHS):
    #     print(f"Epoch {epoch + 1} of {NUM_EPOCHS}")
    #     trainer.train_one_epoch()
    #     trainer.evaluate_one_epoch()


if __name__ == "__main__":
    quick_run(
        CNNModel(4, 2),
        test_size=0.2,
        seed=123
    )
