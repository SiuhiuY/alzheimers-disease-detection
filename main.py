import os
import optuna
import numpy as np
import random
import pickle
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from wandb.integration.keras import WandbMetricsLogger
from src.feature_loader import load_features
from src.dataset import EEGDataset
from src.models.CNN import CNNModel
from cross_validation import CrossValidator
from src.model_trainer import ModelTrainer
import src.util as util

# References:
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# https://discuss.pytorch.org/t/k-fold-cross-validation-with-optuna/182229

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
num_classes = 1  # Binary classification
class_names = [
    k for k, v in sorted(label_map.items(), key=lambda item: item[1])
]

# Define training configuration
RANDOM_SEED = 123
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = torch.optim.Adam
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(45)
])


def run_model(
    model_name, n_splits=5, outer_cv_strategy='loso',
    inner_cv_strategy='sgkf', use_class_weights=False,
    seed=RANDOM_SEED
):
    """Run the model training and evaluation pipeline."""

    # Set random seeds
    util.reproducability(seed)

    # Load data
    print("Loading features and labels...")
    features, labels, subjects = load_features(
        label_map, band=BAND
    )
    print("Data loaded.")
    print("Classification task:", label_map)

    # Calculate class weights
    if use_class_weights:
        class_weights = util.calculate_class_weights(labels).to(DEVICE)
        CRITERION = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        CRITERION = torch.nn.BCEWithLogitsLoss()

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

    # Initialise a dictitionary to store overall results
    all_results = {
        "meta_data": {
            "model_name": model_name.__class__.__name__,
            "class_names": class_names,
            "label_map": label_map,
            "band": BAND
        },
        "outer_folds": [],
        "subject_metadata": {}
    }

    # Model checkpoint path
    result_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", model_name.__class__.__name__, BAND, "_".join(class_names))
    os.makedirs(result_dir, exist_ok=True)

    for outer_fold, train_val_idx, test_idx in outer_cv.cv_loop():
        print(f"\n--- Outer Fold {outer_fold + 1} ---")

        # # Inner loop to split train/val data into training and validation sets
        # # Retrieve indices from the train/val set for inner CV
        # inner_cv = CrossValidator(
        #     features=features[train_val_idx],
        #     labels=labels[train_val_idx],
        #     subjects=subjects[train_val_idx],
        #     cv_strategy=inner_cv_strategy,
        #     n_splits=n_splits,
        #     shuffle=True,
        #     random_state=seed
        # )

        # For each outer fold, split train and val dataset and initiate
        # an Optuna hyperparameter optimisation study
        train_val_set = EEGDataset(
            features[train_val_idx],
            labels[train_val_idx],
            subjects[train_val_idx],
            transform=train_transform
        )
        test_set = EEGDataset(
            features[test_idx],
            labels[test_idx],
            subjects[test_idx],
            transform=None
        )

        train_val_loader = DataLoader(
            train_val_set,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        test_subject_id = subjects[test_idx]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"Outer_Fold_{outer_fold + 1}_Study",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )

        study.optimize(
            lambda trial: objective(
                trial, inner_cv, features, labels,
                subjects, model_name, train_transform
            ),
            n_trials=7)
        
        # Retrieve the best model from this study
        best_trial = study.best_trial
        best_score = best_trial.value
        best_params = best_trial.params

        print(f"Best trial for Outer Fold {outer_fold + 1}:")
        print(f"  Value: {best_score}")
        print("   Params: ")
        for key, value in best_params.items():
            print(f"    {key}: {value}")


        # for inner_fold, train_idx, val_idx in inner_cv.cv_loop():
        #     print(f"  Inner Fold {inner_fold + 1} of {inner_cv.n_splits}")
        #     train_set = EEGDataset(
        #         features[train_idx],
        #         labels[train_idx],
        #         subjects[train_idx],
        #         is_binary=True,
        #         transform=train_transform  # data augmentation for training set
        #     )
        #     train_loader = DataLoader(
        #         train_set,
        #         batch_size=BATCH_SIZE,
        #         shuffle=True
        #     )
        #     val_set = EEGDataset(
        #         features[val_idx],
        #         labels[val_idx],
        #         subjects[val_idx],
        #         is_binary=True,
        #         transform=None
        #     )

        #     val_loader = DataLoader(
        #         val_set,
        #         batch_size=BATCH_SIZE,
        #         shuffle=False
        #     )

        #     # Initialise model
        #     model = model_name.to(DEVICE)

        #     trainer = ModelTrainer(
        #         model, train_loader, val_loader,
        #         OPTIMIZER(model.parameters(), lr=LEARNING_RATE),
        #         CRITERION, DEVICE, threshold=0.5
        #     )

        #     # Training model
        #     history = trainer.fit(NUM_EPOCHS)

        # Retrain best model on the entire train/val set
        best_lr = best_params["learning_rate"]
        optimizer_name = best_params["optimizer"]
        best_optimizer = util.get_optimizer(
            optimizer_name, model_name.parameters(), best_lr
        )
        new_model = 
        model = model_name.to(DEVICE)
        trainer = ModelTrainer(model, train_val_loader, test_loader,
                               best_optimizer(model.parameters(), lr=best_lr),
                               CRITERION, DEVICE, threshold=0.5)

        # Train on the entire train/val set
        trainer.fit(NUM_EPOCHS)


        # Save the best model
        model_filepath = os.path.join(
            result_dir, f"best_model_outer_fold_{outer_fold + 1}.pt"
        )
        torch.save(model.state_dict(), model_filepath)

        # Reload the best model

        # Evaluate on the test set
        test_metrics = trainer.calculate_metrics(new_model, test_loader)

        print(f"Test Metrics for Outer Fold {outer_fold + 1}:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Store outer fold results
        fold_result = {
            "outer_fold": outer_fold + 1,
            "test_subject_id": test_subject_id,
            "best_params": best_params,
            "best_epochs": trainer.best_epoch,
            "model_filepath": model_filepath,

            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_auc": test_metrics["auc"],
            "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),

            "true_labels": test_metrics["y_true"].tolist(),
            "pred_probs": test_metrics["y_pred_probs"].flatten().tolist(),
            "pred_labels": test_metrics["y_pred"].tolist()
        }

        all_results["outer_folds"].append(fold_result)

    # Save overall results
    with open(os.path.join(result_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print("Nested cross-validation completed. Results saved.")


if __name__ == "__main__":
    run_model(
        CNNModel(4, 1),
        n_splits=5,
        outer_cv_strategy='loso',
        inner_cv_strategy='sgkf',
        use_class_weights=True,
        seed=RANDOM_SEED
    )
