import os
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from src.feature_loader import load_features
from src.dataset import EEGDataset
from src.models.CNN import CNNModel
from cross_validation import CrossValidator
from src.util import reproducability
from src.model_trainer import ModelTrainer

# Reference:
# https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data configuration
LABEL_PATH = "data/participants.tsv"
LABEL_MAP = {"A": 0, "F": 1, "C": 2}
BAND = "alpha"

# Define training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


def quick_run(model_name, test_size=0.2, seed=123):
    """Run the model training and evaluation pipeline."""

    # Set random seeds
    reproducability(seed)

    # Load data
    features, labels, subjects = load_features(
        LABEL_PATH, LABEL_MAP, band=BAND
    )

    # Create dataset
    full_dataset = EEGDataset(features, labels, subjects)

    # Create a single train/test split stratified by class and grouped by
    # subject.
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

    train_subset = torch.utils.data.Subset(full_dataset, train_ind)
    test_subset = torch.utils.data.Subset(full_dataset, test_ind)

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
        CNNModel(3, 3),
        test_size=0.2,
        seed=123
    )
