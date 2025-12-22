import numpy as np
import copy
import torch
import os

# Reference:
# https://github.com/Bjarten/early-stopping-pytorch/blob/main/early_stopping_pytorch/early_stopping.py
# https://github.com/souvlasvegas/EEG-DICE-net/blob/main/machine_learning/early_stopping.py
# https://github.com/keras-team/keras/blob/v3.12.0/keras/src/callbacks/early_stopping.py#L8


class EarlyStopping:
    """Early stops the training if training metric doesn't improve
    after a given patience.

    Args:
        patience (int): How many epochs to wait after the last time
            the monitored metric improved. Default: 5
        min_delta (float): Minimum change in the monitored metric to
            qualify as an improvement. Default: 0
        restore_best_weights (bool): Whether to restore model weights
            from the epoch with the best value of the monitored metric.
        verbose (bool): If True, prints progress messages.

    Attributes:
        counter: number of epochs since last improvement.
        best_loss: best recorded validation loss.
        best_model: model state dict with the best recorded validation loss.
        best_epoch: epoch number of the best recorded validation loss.

    Returns:
        bool: True if training should stop, False otherwise.
    """
    def __init__(self, patience=5, min_delta=0,
                 restore_best_weights=True,
                 start_from_epoch=0,
                 verbose=0):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        """Call method to check if training should stop.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Current model.
            epoch (int, optional): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if epoch is None:
            raise ValueError("Epoch number must be provided.")

        if np.isnan(val_loss):
            # Prevent corruption caused by numerical instability
            if self.verbose:
                print("Warning: Validation loss is NaN. Ignoring this epoch.")
            return False

        # Skip early stopping check until start_from_epoch
        if epoch < self.start_from_epoch:
            return False

        # First epoch initialisation
        if self.best_loss is None:
            # Initialize best_loss on first epoch
            self.best_loss = val_loss
            # state_dict: a dictionary containing model's parameters
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            if self.verbose:
                print(f"Epoch {epoch}: Initial loss = {val_loss:.6f}")

        # Check for improvement
        elif self.best_loss - val_loss > self.min_delta:
            # Significant improvement
            if self.verbose:
                print(f"Epoch {epoch}: Loss improved from {self.best_loss:.6f}"
                      f" to {val_loss:.6f}.")
            self.best_loss = val_loss
            # Overwrite with new weights
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0  # reset counter

        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                print(f"Epoch {epoch}: No improvement in loss for "
                      f"{self.counter} epoch(s).")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}. "
                          f"Best loss: {self.best_loss:.6f} "
                          f"at epoch {self.best_epoch}.")
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                    if self.verbose:
                        print("Model weights restored to best epoch.")
                return True

        return False
