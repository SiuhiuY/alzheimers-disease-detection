import torch
from torch.utils.data import DataLoader
from src.feature_loader import load_features
from src.dataset import EEGDataset
from src.models.CNN import CNNModel
from cross_validation import CrossValidator
from src.model_trainer import ModelTrainer
import src.util as util


# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define data configuration
LABEL_PATH = "data/participants.tsv"
AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
AD_CN = {"A": 1, "C": 0}
FTD_CN = {"F": 1, "C": 0}
AD_FTD = {"A": 1, "F": 0}
label_map = AD_CN
BAND = "alpha"

# Define training configuration
RANDOM_SEED = 123
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


def run_model(
    model_name, n_splits=5, outer_cv_strategy='loso',
    inner_cv_strategy='sgkf', seed=RANDOM_SEED
):
    """Run the model training and evaluation pipeline."""

    # Set random seeds
    util.reproducability(seed)

    # Load data
    features, labels, subjects = load_features(
        label_map, band=BAND
    )

    # Create dataset
    full_dataset = EEGDataset(features, labels, subjects)

    # Nested Cross-Validation setup
    outer_cv = CrossValidator(
        full_dataset=full_dataset,
        features=features,
        labels=labels,
        subjects=subjects,
        cv_strategy=outer_cv_strategy,
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    all_metrics = []

    # Outer loop for model evaluation
    for fold, (train_subset, test_subset) in outer_cv.outer_loop():
        print(f"Starting Fold {fold + 1} of {outer_cv.n_splits}")

        # Create DataLoaders for training and testing
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=BATCH_SIZE, shuffle=False
        )

        # Inner loop to split training data into training and validation sets

        # Retrieve indices from the training set for inner CV
        train_indices = train_subset.indices

        # Extract features, labels, and subjects for the training subset
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        train_subjects = subjects[train_indices]

        # Initialise inner cross-validator
        inner_cv = CrossValidator(
            full_dataset=train_subset,
            features=train_features,
            labels=train_labels,
            subjects=train_subjects,
            cv_strategy=inner_cv_strategy,
            n_splits=n_splits
        )
        for i, (train_idx, val_idx) in enumerate(inner_cv.inner_loop()):
            print(f"  Inner Fold {i + 1} of {inner_cv.n_splits}")
            train_loader = DataLoader(
                train_subset[train_idx], batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                train_subset[val_idx], batch_size=BATCH_SIZE, shuffle=False
            )

            # Initialise model
            model = model_name().to(DEVICE)


if __name__ == "__main__":
    run_model(
        CNNModel,
        n_splits=5,
        outer_cv_strategy='loso',
        inner_cv_strategy='sgkf',
        seed=123
    )
