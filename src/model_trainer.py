
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    def __init__(
        self, model, train_loader, test_loader, optimizer, criterion, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.accuracy = Accuracy(task="multiclass", num_classes=3)

    def train_one_epoch(self):
        # Set model to training mode
        self.model.train()

        # Initialise running loss
        running_loss = 0.0
        acc = self.accuracy

        # Training in batches
        for features, labels, _ in self.train_loader:
            # Move data to device
            features, labels = features.to(self.device), labels.to(self.device)
            # Zero the gradients before each new batch
            self.optimizer.zero_grad()
            # Forward pass to make predictions
            # output shape: (batch_size, num_classes)
            outputs = self.model(features)
            # Compute loss between predictions and true labels
            loss = self.criterion(outputs, labels)
            # Compute gradients for every parameter by backpropagation
            loss.backward()
            # Update model weights based on computed gradients
            self.optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            # Update accuracy metric
            acc.update(preds, labels)

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(self.train_loader)
        accuracy = acc.compute()
        acc.reset()
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"Training Accuracy: {accuracy:.4f}")

    def evaluate_one_epoch(self):
        # Set model to evaluation mode
        self.model.eval()

        # Initialise accuracy metric
        running_loss = 0.0
        acc = self.accuracy
        # Disable gradient backpropagation for evaluation
        with torch.no_grad():
            for features, labels, _ in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                acc.update(preds, labels)

        epoch_loss = running_loss / len(self.test_loader)
        accuracy = acc.compute()
        acc.reset()
        print(f"Validation Loss: {epoch_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} of {num_epochs}")
            self.train_one_epoch()
            self.evaluate_one_epoch()


if __name__ == "__main__":
    print("ModelTrainer module")
