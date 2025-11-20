import torch
import torchvision.transforms as T
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
CRITERION = torch.nn.BCEWithLogitsLoss()  # combines a Sigmoid layer and the BCELoss in one single class
OPTIMIZER = torch.optim.Adam
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(45)
])


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

    all_metrics = []

    # Outer loop for model evaluation
    outer_cv = CrossValidator(
        features=features,
        labels=labels,
        subjects=subjects,
        cv_strategy=outer_cv_strategy,
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for outer_fold, train_val_idx, test_idx in outer_cv.cv_loop():
        print(f"Starting Fold {outer_fold + 1} of {outer_cv.n_splits}")
        # Create DataLoaders for train/val and test sets
        # train_val_set = EEGDataset(
        #     features[train_val_idx],
        #     labels[train_val_idx],
        #     subjects[train_val_idx]
        # )
        test_set = EEGDataset(
            features[test_idx],
            labels[test_idx],
            subjects[test_idx],
            transform=None
        )

        # train_val_loader = DataLoader(
        #     train_val_set,
        #     batch_size=BATCH_SIZE,
        #     shuffle=True  # shuffle for more diverse batches
        # )
        test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False  # no need to shuffle for evaluation
        )

        # Inner loop to split train/val data into training and validation sets
        # Retrieve indices from the train/val set for inner CV
        inner_cv = CrossValidator(
            features=features[train_val_idx],
            labels=labels[train_val_idx],
            subjects=subjects[train_val_idx],
            cv_strategy=inner_cv_strategy,
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        for inner_fold, train_idx, val_idx in inner_cv.cv_loop():
            print(f"  Inner Fold {inner_fold + 1} of {inner_cv.n_splits}")
            train_set = EEGDataset(
                features[train_idx],
                labels[train_idx],
                subjects[train_idx],
                transform=train_transform  # data augmentation for training set
            )
            train_loader = DataLoader(
                train_set,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            val_set = EEGDataset(
                features[val_idx],
                labels[val_idx],
                subjects[val_idx],
                transform=None
            )

            val_loader = DataLoader(
                val_set,
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            # Initialise model
            model = model_name().to(DEVICE)

            trainer = ModelTrainer(
                model, train_loader, val_loader,
                OPTIMIZER(model.parameters(), lr=LEARNING_RATE),
                CRITERION, DEVICE
            )

            # Training model
            trainer.fit(NUM_EPOCHS)

        # Evaluate on the held-out test set
        test_loss, test_acc = trainer.evaluate_one_epoch()


if __name__ == "__main__":
    run_model(
        CNNModel(4, 2),
        n_splits=5,
        outer_cv_strategy='loso',
        inner_cv_strategy='sgkf',
        seed=123
    )
