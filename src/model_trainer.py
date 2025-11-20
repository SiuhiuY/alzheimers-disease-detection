
import torch
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
)

# Reference:
# https://pub.towardsai.net/improve-your-model-validation-with-torchmetrics-b457d3954dcd


class ModelTrainer:
    """Class to handle model training, validation, and testing."""
    def __init__(
        self, model, train_loader, val_loader, test_loader,
        optimizer, criterion, device, threshold=0.5
    ):
        """Initialise the ModelTrainer.

        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for test data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run the model on.
            threshold (float): Threshold for converting probabilities to
                class labels.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        self.criterion = criterion
        self.optimizer = optimizer

        self.accuracy = Accuracy(task="binary").to(device)
        self.precision = Precision(task="binary").to(device)
        self.recall = Recall(task="binary").to(device)
        self.f1_score = F1Score(task="binary").to(device)
        self.auroc = AUROC(task="binary").to(device)
        self.confusion_matrix = ConfusionMatrix(task="binary").to(device)

    def train_one_epoch(self):
        """Train the model for one epoch.

        Returns:
            float: Average training loss for the epoch.
            float: Training accuracy for the epoch.
        """
        # Set model to training mode
        self.model.train()

        # Initialise running loss
        running_loss = 0.0
        acc = self.accuracy

        # Training in batches
        for features, labels, _ in self.train_loader:
            # Get data from the dataloader and move it to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero the gradients before each new batch
            self.optimizer.zero_grad()
            # Run a forward pass
            # output shape: (batch_size, num_classes)
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
            # Transform into predicted class labels
            preds = (outputs >= self.threshold).float()
            # Update accuracy metric
            acc.update(preds, labels.view(-1, 1))

        # Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / len(self.train_loader)
        accuracy = acc.compute()
        acc.reset()

        return epoch_loss, accuracy

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
        acc = self.accuracy

        # Disable gradient backpropagation for evaluation
        with torch.no_grad():
            for features, labels, _ in self.val_loader:

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels.view(-1, 1))
                running_loss += loss.item()

                preds = (outputs >= self.threshold).float()
                acc.update(preds, labels.view(-1, 1))

        epoch_loss = running_loss / len(self.val_loader)
        accuracy = acc.compute()
        acc.reset()

        # set back to train mode
        self.model.train()

        return epoch_loss, accuracy

    def fit(self, num_epochs):
        """Train and validate the model for a given number of epochs.

        Args:
            num_epochs (int): Number of epochs to train the model.

        Returns:
            dict: Training history containing loss and accuracy for
                training and validation sets.
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate_one_epoch()
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc.item())  # convert tensor to float
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc.item())
        return history

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
            for features, labels, _ in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                pred_probs = outputs.cpu().numpy()
                y_pred_probs.extend(pred_probs)
                preds = (outputs >= self.threshold).float()
                y_preds.extend(preds.cpu().numpy())
                y_labels.extend(labels.cpu().numpy())
        return {
            "y_pred_probs": y_pred_probs,
            "y_pred": y_preds,
            "y_true": y_labels
        }

    def calculate_metrics(self):
        """Calculate evaluation metrics on the test set.

        Returns:
            dict: Dictionary containing accuracy, precision, recall,
                F1-score, confusion matrix, and AUC.
        """
        self.model.eval()
        acc = self.accuracy
        precision = self.precision
        recall = self.recall
        f1 = self.f1_score
        auroc = self.auroc
        confusion_matrix = self.confusion_matrix
        with torch.no_grad():
            for features, labels, _ in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                preds = (outputs >= self.threshold).float()

                acc.update(preds, labels.view(-1, 1))
                precision.update(preds, labels.view(-1, 1))
                recall.update(preds, labels.view(-1, 1))
                f1.update(preds, labels.view(-1, 1))
                confusion_matrix.update(preds, labels.view(-1, 1))
                auroc.update(outputs, labels.view(-1, 1))

        test_acc = acc.compute()
        test_precision = precision.compute()
        test_recall = recall.compute()
        test_f1 = f1.compute()
        test_confusion_matrix = confusion_matrix.compute()
        test_auroc = auroc.compute()

        acc.reset()
        precision.reset()
        recall.reset()
        f1.reset()
        confusion_matrix.reset()
        auroc.reset()

        return {
            "accuracy": test_acc,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "confusion_matrix": test_confusion_matrix,
            "auc": test_auroc
        }
