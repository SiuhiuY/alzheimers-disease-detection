import optuna
import numpy as np
import wandb
import torch
import torch.nn as nn
import src.util as util
from src.model_trainer import ModelTrainer
from src.callback import EarlyStopping
from src.cross_validation import CrossValidator

# Reference:
# https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py
# https://stackoverflow.com/questions/67504503/optuna-pass-dictionary-of-parameters-from-outside


class Objective(object):
    """Objective function for Optuna hyperparameter tuning
    with cross-validation."""

    def __init__(
        self, features, labels, subjects,
        train_idx, val_idx,
        model_builder, train_transform,
        device, num_epochs=20,
        use_class_weights=True
    ):
        """Initialise the Objective.

        Args:
            features (np.ndarray): Feature data.
            labels (np.ndarray): Corresponding labels.
            subjects (np.ndarray): Subject identifiers.
            train_idx (list or np.ndarray): Indices for training samples.
            val_idx (list or np.ndarray): Indices for validation samples.
            model_builder (callable): Function to build the model.
            train_transform (callable): Data augmentation transform
                for training data.
            device (torch.device): Device to run the model on.
            num_epochs (int): Number of epochs to train.
            use_class_weights (bool): Whether to use class weights.
        """
        self.features = features
        self.labels = labels
        self.subjects = subjects
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.model_builder = model_builder
        self.train_transform = train_transform
        self.device = device
        self.num_epochs = num_epochs
        self.use_class_weights = use_class_weights

    def __call__(self, trial):
        """Call method to execute the objective function.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: accuracy on the validation set.
        """
        # Suggest optimiser and learning rate
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "RMSprop"]
        )
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        )

        # Create training and validation datasets
        train_loader, val_loader = util.get_data_loaders(
            self.features, self.labels, self.subjects,
            self.train_idx, self.val_idx,
            train_transform=self.train_transform,
            batch_size=batch_size,
            shuffle=True
        )

        # Build model
        model = self.model_builder(trial).to(self.device)

        # Set up loss function
        criterion = util.get_criterion(
            self.use_class_weights, self.labels[self.train_idx], self.device
            )

        # Set up optimizer
        optimizer = util.get_optimizer(
            optimizer_name, model.parameters(), learning_rate, weight_decay
        )

        # Train the model
        trainer = ModelTrainer(
            model, train_loader, val_loader,
            optimizer, criterion, self.device, threshold=0.5)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=5, min_delta=0.0,
            restore_best_weights=True,
            verbose=0
        )

        # Initialise val accuracy
        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            train_loss, train_acc = trainer.train_one_epoch()
            val_loss, val_acc = trainer.evaluate_one_epoch()

            # Log training and validation metrics to wandb
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

            # Update best validation accuracy
            if val_acc.item() > best_val_acc:
                best_val_acc = val_acc.item()

            # Check for early stopping
            if early_stopping(val_loss, model, epoch):
                break

            # Monitor val_acc for pruning
            trial.report(val_acc.item(), epoch)
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.TrialPruned()

        # Log best accuracy and best epoch to wandb
        wandb.run.summary["best_val_accuracy"] = best_val_acc
        wandb.run.summary["best_epoch"] = early_stopping.best_epoch
        wandb.run.summary["state"] = "completed"
        wandb.finish()

        # Store the best accuracy and best epoch
        trial.set_user_attr("best_val_acc", best_val_acc)
        trial.set_user_attr("best_epoch", early_stopping.best_epoch)

        print(f"Best Validation Accuracy: {best_val_acc:.4f} "
              f"at Epoch {early_stopping.best_epoch}")

        return best_val_acc
