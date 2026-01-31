import optuna
import numpy as np
import copy
import wandb
import torch
import torch.nn as nn
import src.util as util
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        band, class_names, label_map,
        outer_fold, model_name, seed,
        device, num_epochs=50,
        use_class_weights=True,
        best_model_state=None
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
        self.band = band
        self.class_names = class_names
        self.label_map = label_map
        self.outer_fold = outer_fold
        self.model_name = model_name
        self.seed = seed
        self.best_model_state = best_model_state

    def __call__(self, trial):
        """Call method to execute the objective function.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: accuracy on the validation set.
        """
        # Suggest optimiser and learning rate
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam"]
        )
        learning_rate = trial.suggest_float(
            "learning_rate", 1.5e-5, 3e-5, log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", 3e-6, 2e-5, log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size", [128, 256]
        )

        # Intialise wandb for this trial
        wandb_config = {**trial.params, "trial_number": trial.number}
        classes_str = '_'.join(self.class_names)

        wandb.init(
            project=f"EEG_Classification_2D_{self.band}_{classes_str}_v1",
            name=f"Outer_Fold_{self.outer_fold + 1}_Trial_{trial.number}",
            config=wandb_config,
            group=f"Outer_Fold_{self.outer_fold + 1}",
            job_type="tuning",
            reinit=True
        )

        # Build model
        model = self.model_builder(trial).to(self.device)

        # Create training and validation datasets
        train_loader, val_loader = util.get_data_loaders(
            self.features, self.labels, self.subjects,
            self.train_idx, self.val_idx,
            train_transform=self.train_transform,
            batch_size=batch_size,
            shuffle=True
        )

        # Set up optimizer and loss function
        optimizer = util.get_optimizer(
            optimizer_name, model.parameters(), learning_rate, weight_decay
        )
        criterion = util.get_criterion(
            self.use_class_weights, self.labels[self.train_idx], self.device
            )

        # Set up early stopping and learning rate scheduler
        early_stopping = EarlyStopping(
            patience=15,
            restore_best_weights=True,
            start_from_epoch=5,
            verbose=0
        )
        reduce_lr = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=5, min_lr=1e-6, verbose=False
            )

        # Train the model
        trainer = ModelTrainer(
            model, train_loader, val_loader,
            criterion, self.device, optimizer, threshold=0.5
            )

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

            # Monitor validation loss for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.TrialPruned()

            # Adjust learning rate based on validation loss
            reduce_lr.step(val_loss)

            # Check for early stopping
            if early_stopping(val_loss, model, epoch):
                break

        # Get the number of trained epochs and the best epoch
        trained_epochs = epoch + 1
        best_epoch = early_stopping.best_epoch

        # Evaluate the model on the validation set using the best model weights
        model.load_state_dict(early_stopping.best_model)
        trial_val_loss, trial_val_accuracy = trainer.evaluate_one_epoch()
        print(f"Trial {trial.number} - "
              f"Val Loss: {trial_val_loss:.4f}, "
              f"Val Accuracy: {trial_val_accuracy:.4f} ")

        # Check if this is the best trial so far
        trial_best_loss = early_stopping.best_loss
        if trial_best_loss < self.best_model_state["best_val_loss"]:
            # Overwrite best model state with the current best
            self.best_model_state["best_val_loss"] = trial_best_loss
            torch.save(
                early_stopping.best_model, self.best_model_state["path"]
            )
            trial.set_user_attr(
                "best_model_path", self.best_model_state["path"]
            )

        # Set user attributes for the trial
        trial.set_user_attr("trained_epochs", trained_epochs)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("val_loss", trial_val_loss)
        trial.set_user_attr("val_accuracy", trial_val_accuracy)

        # Log results to wandb
        wandb.run.summary["trained_epochs"] = trained_epochs
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["val_loss"] = trial_val_loss
        wandb.run.summary["val_accuracy"] = trial_val_accuracy
        wandb.run.summary["state"] = "completed"
        wandb.finish()

        return trial_val_loss
