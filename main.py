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
from src.feature_loader import load_features
from src.dataset import EEGDataset
from src.models.optuna_cnn import OptunaCNN
from src.cross_validation import CrossValidator
from src.model_trainer import ModelTrainer
from src.model_tuner import Objective
import src.util as util

# References:
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# https://discuss.pytorch.org/t/k-fold-cross-validation-with-optuna/182229


def run_model(
    model_name, label_map,
    n_splits=None, test_size=0.2, n_epochs=20, band="alpha",
    outer_cv_strategy='loso', inner_cv_strategy=None,
    use_class_weights=True,
    train_transform=None,
    seed=123
):
    """Run the model training and evaluation pipeline."""

    # Set random seeds
    util.reproducability(seed)

    # Define device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    class_names = [
        k for k, v in sorted(label_map.items(), key=lambda item: item[1])
    ]

    # Load data
    print("Loading features and labels...")
    features, labels, subjects = load_features(
        label_map=label_map, band=band
    )
    print("Data loaded.")
    print("Classification task:", label_map)

    # Outer loop for model evaluation
    outer_cv = CrossValidator(
        features=features,
        labels=labels,
        subjects=subjects,
        cv_strategy=outer_cv_strategy,
        n_splits=n_splits,
        test_size=test_size,
        shuffle=True,
        random_state=seed
    )

    # Initialise a dictitionary to store overall results
    all_results = {
        "meta_data": {
            "model_name": model_name.__name__,
            "class_names": class_names,
            "label_map": label_map,
            "band": band,
        },
        "outer_folds": [],
        "subject_metadata": {}
    }

    # Model checkpoint path
    result_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", model_name.__name__, band, "_".join(class_names))
    os.makedirs(result_dir, exist_ok=True)

    for outer_fold, train_val_idx, test_idx in outer_cv.cv_loop():
        print(f"\n--- Outer Fold {outer_fold + 1} ---")
        print(f"Train/Val size: {len(train_val_idx)}, "
              f"Test size: {len(test_idx)}")
        print(f"Test subject IDs: {np.unique(subjects[test_idx])}")

        # Inner loop for hyperparameter tuning
        inner_cv = CrossValidator(
            features=features[train_val_idx],
            labels=labels[train_val_idx],
            subjects=subjects[train_val_idx],
            cv_strategy=inner_cv_strategy,
            n_splits=n_splits,
            test_size=test_size,
            shuffle=True,
            random_state=seed
        )

        if inner_cv_strategy is None:
            # Single train/val split
            _, train_idx, val_idx = next(inner_cv.cv_loop())

            # Define model builder function for Objective
            def model_builder(trial):
                C = features.shape[1]
                H = features.shape[2]
                W = features.shape[3]
                input_shape = (C, H, W)

                return model_name(trial, input_shape=input_shape, num_classes=1)

            # Create optuna study for hyperparameter tuning
            study = optuna.create_study(
                direction="maximize",
                study_name=f"Outer_Fold_{outer_fold + 1}_Study",
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )

            objective = Objective(
                features[train_val_idx],
                labels[train_val_idx],
                subjects[train_val_idx],
                train_idx, val_idx,
                model_builder,
                train_transform,
                band, class_names, label_map,
                outer_fold, model_name, seed,
                DEVICE,
                num_epochs=n_epochs,
                use_class_weights=use_class_weights,
            )

            study.optimize(
                lambda trial: objective(trial), n_trials=10)

            # Retrieve the best model from this study
            best_trial = study.best_trial
            best_score = best_trial.value
            best_params = best_trial.params
            best_batch_size = best_params["batch_size"]
            best_epochs = best_trial.user_attrs["best_epoch"]

            criterion = util.get_criterion(
                use_class_weights, labels[train_val_idx], DEVICE
            )

            print(f"Best trial for Outer Fold {outer_fold + 1}:")
            print(f"  Value: {best_score}")
            print("   Params: ")
            for key, value in best_params.items():
                print(f"    {key}: {value}")

            # Save the best model from this study
            best_model_state = best_trial.user_attrs["best_model_state_dict"]
            model_filepath = os.path.join(
                result_dir, f"best_model_{outer_fold + 1}.pt"
            )
            torch.save(best_model_state, model_filepath)

            # Retrain the model on the entire train/val set for evaluation
            # Split train_val and test sets
            train_val_loader, test_loader = util.get_data_loaders(
                features,
                labels,
                subjects,
                train_val_idx, test_idx,
                train_transform=train_transform,
                batch_size=best_batch_size, shuffle=True
            )
            test_subject_id = subjects[test_idx]

            # Load the best model architecture
            final_model = model_builder(best_trial).to(DEVICE)
            final_model.load_state_dict(best_model_state)

            best_lr = best_params["learning_rate"]
            optimizer_name = best_params["optimizer"]
            best_optimizer = util.get_optimizer(
                optimizer_name, final_model.parameters(),
                best_lr, weight_decay=best_params["weight_decay"]
            )

            # Retrain the model
            trainer = ModelTrainer(final_model, train_val_loader, test_loader,
                                   optimizer=best_optimizer,
                                   criterion=criterion, device=DEVICE,
                                   threshold=0.5)

            # Train on the entire train/val set
            trainer.fit(best_epochs)

            # Evaluate on the test set
            test_metrics = trainer.calculate_metrics()
            predictions = trainer.predict()

            print(f"Test Metrics for Outer Fold {outer_fold + 1}:")
            for metric_name, metric_value in test_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

            # Store outer fold results
            fold_result = {
                "outer_fold": outer_fold + 1,
                "best_trial_number": best_trial.number,
                "test_subject_id": test_subject_id,
                "best_params": best_params,
                "best_epochs": best_epochs,
                "model_filepath": model_filepath,

                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1_score": test_metrics["f1_score"],
                "test_auc": test_metrics["auc"],
                "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),

                "true_labels": predictions["y_true"],
                "pred_probs": predictions["y_pred_probs"].flatten().tolist(),
                "pred_labels": predictions["y_pred"].tolist()
            }

            all_results["outer_folds"].append(fold_result)

        else:
            # Inner CV
            pass

    # Save overall results
    with open(os.path.join(result_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    # Calculate overall statistics
    all_accuracies = [
        fold["test_accuracy"] for fold in all_results["outer_folds"]
    ]
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)

    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

    print("Nested cross-validation completed. Results saved.")


if __name__ == "__main__":

    # Define data configuration
    LABEL_PATH = "data/participants.tsv"
    AD_FTD_CN = {"A": 0, "F": 1, "C": 2}
    AD_CN = {"A": 1, "C": 0}
    FTD_CN = {"F": 1, "C": 0}
    AD_FTD = {"A": 1, "F": 0}
    label_map = AD_CN
    band = "alpha"
    num_classes = 1  # Binary classification

    # Define training configuration
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(45)
    ])
    RANDOM_SEED = 123
    n_epochs = 20

    # Run the model training
    run_model(
        OptunaCNN,
        label_map=label_map,
        n_splits=None,
        test_size=0.2,
        n_epochs=n_epochs,
        band=band,
        outer_cv_strategy='loso',
        inner_cv_strategy=None,  # inner single train / val split
        use_class_weights=True,
        train_transform=train_transform,
        seed=RANDOM_SEED
    )
