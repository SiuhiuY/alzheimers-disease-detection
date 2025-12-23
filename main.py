import os
import optuna
import numpy as np
import random
import pickle
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as T
from src.feature_loader import load_features
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

    # Outer Leave-One Subject Out cross-validation loop
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
            _, inner_train_idx, inner_val_idx = next(inner_cv.cv_loop())

            # Map back to original indices
            train_idx = train_val_idx[inner_train_idx]
            val_idx = train_val_idx[inner_val_idx]
            print(f"Inner Train size: {len(train_idx)}, "
                  f"Val size: {len(val_idx)}")

            # Sanity check
            if not np.array_equal(subjects[train_val_idx][inner_train_idx],
                                  subjects[train_idx]):
                raise ValueError("Index mapping error in inner CV.")

            # Define model builder function for Objective
            def model_builder(trial):
                C = features.shape[1]
                H = features.shape[2]
                W = features.shape[3]
                input_shape = (C, H, W)

                return model_name(trial, input_shape=input_shape, num_classes=1)

            # Initialise best model state for this study
            best_model_state = {
                "best_val_loss": float('inf'),
                "path": os.path.join(
                    result_dir, f"best_model_outer_fold_{outer_fold + 1}.pt"
                )
            }

            # Create optuna study for hyperparameter tuning
            study = optuna.create_study(
                direction="minimize",
                study_name=f"Outer_Fold_{outer_fold + 1}_Study",
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                   n_warmup_steps=5)
            )

            objective = Objective(
                features,
                labels,
                subjects,
                train_idx, val_idx,
                model_builder,
                train_transform,
                band, class_names, label_map,
                outer_fold, model_name, seed,
                DEVICE,
                num_epochs=n_epochs,
                use_class_weights=use_class_weights,
                best_model_state=best_model_state
            )

            study.optimize(
                lambda trial: objective(trial), n_trials=10)

            # Save the study
            study_filepath = os.path.join(
                result_dir, f"study_outer_fold_{outer_fold + 1}.pkl")
            with open(study_filepath, "wb") as f:
                pickle.dump(study, f)

            # Retrieve the best trial from this study
            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = best_trial.user_attrs["val_accuracy"]
            best_model_path = best_model_state["path"]
            best_epochs = best_trial.user_attrs["best_epoch"]
            trained_epochs = best_trial.user_attrs["trained_epochs"]

            print(f"Best trial for Outer Fold {outer_fold + 1}:")
            print(f"Best validation accuracy: {best_score:.4f}")
            print("Best hyperparameters: ")
            for key, value in best_params.items():
                print(f"    {key}: {value}")

            # Reload best model state dict
            best_model = model_builder(best_trial).to(DEVICE)
            best_model.load_state_dict(torch.load(best_model_path))

            # Retrieve best hyperparameters
            best_batch_size = best_params["batch_size"]
            criterion = util.get_criterion(
                use_class_weights, labels[train_val_idx], DEVICE
            )

            # Evaluate on the test set
            train_val_loader, test_loader = util.get_data_loaders(
                features,
                labels,
                subjects,
                train_val_idx, test_idx,
                train_transform=train_transform,
                batch_size=best_batch_size, shuffle=True
            )
            test_subject_id = subjects[test_idx]

            # Create trainer for best model
            best_trainer = ModelTrainer(best_model,
                                        train_val_loader,
                                        test_loader,
                                        criterion=criterion,
                                        device=DEVICE,
                                        optimizer=None,
                                        threshold=0.5
                                        )

            # Evaluate on the test set
            _, test_accuracy = best_trainer.evaluate_one_epoch()
            predictions = best_trainer.predict()
            y_pred_probs = predictions["y_pred_probs"]
            y_pred = predictions["y_pred"]
            y_true = predictions["y_true"]
            print(f"Outer Fold {outer_fold + 1} Test Accuracy: "
                  f"{test_accuracy:.4f}")

            # Store outer fold results
            fold_result = {
                "outer_fold": outer_fold + 1,
                "best_trial_number": best_trial.number,
                "test_subject_id": test_subject_id,
                "best_params": best_params,
                "best_epochs": best_epochs,
                "trained_epochs": trained_epochs,
                "model_filepath": best_model_path,
                "test_accuracy": test_accuracy.item(),
                "true_labels": y_true,
                "pred_probs": y_pred_probs,
                "pred_labels": y_pred
            }

            all_results["outer_folds"].append(fold_result)

        else:
            # Inner CV
            pass

    # Save overall results
    with open(os.path.join(result_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print("LOSO cross-validation completed. Results saved.")


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
    n_epochs = 50

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
        use_class_weights=False,
        train_transform=train_transform,
        seed=RANDOM_SEED
    )
