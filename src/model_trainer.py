
import torch
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
)

# Reference:
# https://pub.towardsai.net/improve-your-model-validation-with-torchmetrics-b457d3954dcd
# https://stackoverflow.com/questions/64002566/bcewithlogitsloss-trying-to-get-binary-output-for-predicted-label-as-a-tensor
# https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/20


class ModelTrainer:
    """Class to handle model training, validation, and testing."""
    def __init__(
        self, model, train_loader, eval_loader,
        criterion, device, optimizer=None, threshold=0.5
    ):
        """Initialise the ModelTrainer.

        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            eval_loader (DataLoader): DataLoader for validation / test data.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run the model on.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            threshold (float): Threshold for converting probabilities to
                class labels.
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.threshold = threshold
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_accuracy = Accuracy(task="binary").to(device)
        self.eval_accuracy = Accuracy(task="binary").to(device)

    def train_one_epoch(self):
        """Train the model for one epoch.

        Returns:
            float: Average training loss for the epoch.
            float: Training accuracy for the epoch.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be provided for training.")

        # Set model to training mode
        self.model.train()

        # Initialise running loss
        running_loss = 0.0
        acc = self.train_accuracy

        # Training in batches
        for features, labels, _ in self.train_loader:
            # Get data from the dataloader and move it to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero the gradients before each new batch
            self.optimizer.zero_grad()
            # Run a forward pass
            # output shape: (batch_size, num_classes)
            # output in raw logits which matchs BCEWithLogitsLoss
            outputs = self.model(features)
            # Compute loss between predictions and true labels
            # Reshape labels from (batch_size,) to (batch_size, 1)
            # to match outputs tensor shape
            loss = self.criterion(outputs, labels.view(-1, 1))
            # Compute gradients for every parameter by backpropagation
            loss.backward()
            # Update model weights based on computed gradients
            self.optimizer.step()

            # Update running loss
            running_loss += loss.item()
            # Apply sigmoid to get predicted probabilities
            pred_probs = torch.sigmoid(outputs)
            # Transform into predicted class labels
            preds = (pred_probs >= self.threshold).float()
            # Update accuracy metric
            acc.update(preds, labels.view(-1, 1))

        # Compute average loss and accuracy for the epoch
        train_loss = running_loss / len(self.train_loader)
        train_acc = acc.compute()
        acc.reset()

        return train_loss, train_acc

    def evaluate_one_epoch(self):
        """Evaluate the model on the validation set for one epoch.

        Returns:
            float: Average validation loss for the epoch.
            float: Validation accuracy for the epoch.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialise test metric
        running_loss = 0.0
        acc = self.eval_accuracy

        # Disable gradient backpropagation for evaluation
        with torch.no_grad():
            for features, labels, _ in self.eval_loader:

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels.view(-1, 1))
                running_loss += loss.item()

                pred_probs = torch.sigmoid(outputs)
                preds = (pred_probs >= self.threshold).float()
                acc.update(preds, labels.view(-1, 1))

        eval_loss = running_loss / len(self.eval_loader)
        eval_acc = acc.compute()
        acc.reset()

        return eval_loss, eval_acc

    def predict(self):
        """Generate predictions on the test set.

        Returns:
            dict: Dictionary containing predicted probabilities,
                predicted class labels, and true labels.
        """
        self.model.eval()
        y_pred_probs = []
        y_preds = []
        y_labels = []
        with torch.no_grad():
            for features, labels, _ in self.eval_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                pred_probs = torch.sigmoid(outputs)
                preds = (pred_probs >= self.threshold).float()

                # Flatten and move to CPU to save the results
                y_pred_probs.extend(pred_probs.view(-1).cpu().numpy())
                y_preds.extend(preds.view(-1).cpu().numpy())
                y_labels.extend(labels.view(-1).cpu().numpy())

        return {
            "y_pred_probs": y_pred_probs,
            "y_pred": y_preds,
            "y_true": y_labels
        }
