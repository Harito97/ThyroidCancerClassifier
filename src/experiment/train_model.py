import torch
import numpy as np
from sklearn.metrics import f1_score
import json
import wandb


def train_model(
    train_loader=None,
    valid_loader=None,
    model=None,  # model structure (& weights)
    device=None,
    criterion=None,
    optimizer=None,
    num_epoch=100,
    patience=30,
    model_destination=".",
    model_name="model",
):
    """
    Trains a classification model for a specified number of epochs and validates it.

    Parameters:
    train_loader (DataLoader): The DataLoader for training data.
    valid_loader (DataLoader): The DataLoader for validation data.
    model (nn.Module): The model to be trained.
    device (torch.device): The device to train on.
    criterion (nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer for the model parameters.
    num_epoch (int): The number of epochs to train for.
    patience (int): The number of epochs to wait for improvement before stopping training.
    model_destination (str): The directory to save the model to.
    model_name (str): The name of the model.

    Returns:
    None
    """
    if train_loader is None or valid_loader is None:
        print("No data loader is provided")
        return
    if model is None or device is None:
        print("No model or device is provided")
        return
    if criterion is None or optimizer is None:
        print("No criterion or optimizer is provided")
        return

    # Initialize W&B
    wandb.init(
        project="ThyroidCancer",
        entity="harito",
        config={
            "num_epoch": num_epoch,
            "patience": patience,
            "learning_rate": optimizer.defaults["lr"],
            "model_name": model_name,
        },
    )

    print("Moving model, criterion, optimizer to device ...")
    model = model.to(device)
    criterion = criterion.to(device)

    print("Taking model_destination, model_name and history_file_path ...")
    model_destination = (
        model_destination[:-1] if model_destination[-1] == "/" else model_destination
    )
    model_name = model_name.split(".")[0]
    history_file_path = f"{model_destination}/{model_name}_history.json"

    print("Initializing history ...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    print("Training classification model...")
    # best_loss = float("inf")
    best_f1 = -1
    patience_counter = 0

    for epoch in range(num_epoch):
        # Set model in training mode
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        # print(f"Epoch {epoch+1}/{num_epoch}")
        total_batches = len(train_loader)  # Get total number of batches
        for i, (images, labels) in enumerate(train_loader):
            progress = (i + 1) / total_batches * 100  # Calculate progress
            print(f"\rProgress: {progress:.2f}%", end="")

            # Load data of 1 batch to device
            images, labels = images.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.view(-1).cpu().numpy())
            train_targets.extend(labels.view(-1).cpu().numpy())

        # Calculate metrics after a epoch in training set
        # print("\nCalculating metrics in training set ...")
        train_loss = running_loss / len(train_loader)
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_f1 = f1_score(train_targets, train_preds, average="weighted")

        # Validation loop
        # Set model in evaluation mode
        model.eval()
        # print(f"\nStart validation at epoch {epoch + 1} ...")
        val_running_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.view(-1).cpu().numpy())
                val_targets.extend(labels.view(-1).cpu().numpy())

        # Calculate metrics sau epoch này trên tập validation
        val_loss = val_running_loss / len(valid_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # print(
        #     f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Train F1: {train_f1:.6f}\n"
        #     f"Val   Loss: {val_loss:.6f}, Val   Acc: {val_acc:.6f}, Val   F1: {val_f1:.6f}"
        # )

        # Log metrics to W&B
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "epoch": epoch + 1,
            }
        )

        # After the training loop and any early stopping logic
        with open(history_file_path, "w") as history_file:
            json.dump(history, history_file)
        # print("Saved last history at epoch", epoch + 1)

        # Checkpoint
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(
        #         model.state_dict(), f"{model_destination}/best_{model_name}_model.pt"
        #     )
        #     print("Saved **best model** at epoch", epoch + 1)
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     torch.save(
        #         model.state_dict(), f"{model_destination}/last_{model_name}_model.pt"
        #     )
        #     print("Saved last model at epoch", epoch + 1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                model.state_dict(), f"{model_destination}/best_{model_name}_model.pt"
            )
            print("Saved **best model** at epoch", epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1
            torch.save(
                model.state_dict(), f"{model_destination}/last_{model_name}_model.pt"
            )
            # print("Saved last model at epoch", epoch + 1)

        # Early stopping
        if patience_counter >= patience:
            # print("Early stopping")
            break

    # print(f"Training history saved to {history_file_path}")
    print("Training completed")
    wandb.finish()


import os
import json
import torch
import wandb
import numpy as np
from sklearn.metrics import f1_score


class ViTTraining:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        device,
        criterion,
        optimizer,
        num_epochs=100,
        patience=30,
        model_destination=".",
        model_name="model",
        n_layers_to_unfreeze=1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.patience = patience
        self.model_destination = model_destination
        self.model_name = model_name
        self.n_layers_to_unfreeze = n_layers_to_unfreeze
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }
        self.best_f1 = -1
        self.patience_counter = 0

        if not os.path.exists(model_destination):
            os.makedirs(model_destination)

        # Initialize W&B
        wandb.init(
            project="ThyroidCancer",
            entity="harito",
            config={
                "num_epochs": num_epochs,
                "patience": patience,
                "learning_rate": optimizer.defaults["lr"],
                "model_name": model_name,
                "n_layers_to_unfreeze": n_layers_to_unfreeze,
            },
            name="ViT",
        )

    def _set_layer_requires_grad(self, n_layers):
        """
        Set the requires_grad attribute for layers based on the number of layers to unfreeze.
        """
        # Calculate which layers to unfreeze
        start_layer = max(0, self.model.num_layers - n_layers)
        self.model.set_parameter_requires_grad(start_layer, self.model.num_layers)

    def train(self):
        print("Moving model, criterion, optimizer to device ...")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        history_file_path = os.path.join(
            self.model_destination, f"{self.model_name}_history.json"
        )

        print("Training classification model...")
        for epoch in range(self.num_epochs):
            # Gradual unfreezing logic
            n_layers = (epoch + 1) * self.n_layers_to_unfreeze
            self._set_layer_requires_grad(n_layers)

            # Update optimizer for current layer setup
            optimizers = self.model.get_optimizers(n_layers)
            optimizer = torch.optim.AdamW(optimizers, lr=self.model.initial_lr)

            # Training phase
            self.model.train()
            running_loss = 0.0
            train_preds, train_targets = [], []

            total_batches = len(self.train_loader)
            for i, (images, labels) in enumerate(self.train_loader):
                progress = (i + 1) / total_batches * 100
                print(f"\rProgress: {progress:.2f}%", end="")

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.view(-1).cpu().numpy())
                train_targets.extend(labels.view(-1).cpu().numpy())

            train_loss = running_loss / len(self.train_loader)
            train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
            train_f1 = f1_score(train_targets, train_preds, average="weighted")

            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, labels in self.valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.view(-1).cpu().numpy())
                    val_targets.extend(labels.view(-1).cpu().numpy())

            val_loss = val_running_loss / len(self.valid_loader)
            val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
            val_f1 = f1_score(val_targets, val_preds, average="weighted")

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            # Log metrics to W&B
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "epoch": epoch + 1,
                }
            )

            # Save history
            with open(history_file_path, "w") as history_file:
                json.dump(self.history, history_file)

            # Checkpoint
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_destination, f"best_{self.model_name}_model.pt"
                    ),
                )
                print("Saved **best model** at epoch", epoch + 1)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_destination, f"last_{self.model_name}_model.pt"
                    ),
                )

            if self.patience_counter >= self.patience:
                print("Early stopping")
                break

        print("Training completed")
        wandb.finish()
