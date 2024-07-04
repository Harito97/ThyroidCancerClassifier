import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

def train(
    model=None,
    data_version_dir=None,
    num_epoch: int = 100,
    model_destination: str = ".",
    model_name: str = "model",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    # Step 1. Load data and transform data
    print("Loading data...")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        root=f"{data_version_dir}/train", transform=transform
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        root=f"{data_version_dir}/valid", transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Step 2. Prepare model to device
    if model is None:
        print("No model is provided")
        return

    print("Setting up model to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 3. Setup hyperparameters: Prepare loss function and optimizer
    print("Setting up loss function and optimizer...")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_dataset.targets),
        y=train_dataset.targets,
    )

    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Move to the correct device
    class_weights_tensor = class_weights_tensor.to(device)

    # Use the weights in the loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 4. Training loop
    print("Training classification model...")
    best_loss = float("inf")
    patience = 30
    patience_counter = 0

    # Initialize history
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.view(-1).cpu().numpy())
            train_targets.extend(labels.view(-1).cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_f1 = f1_score(train_targets, train_preds, average="weighted")

        # Validation loop
        model.eval()
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

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Train F1: {train_f1}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val F1: {val_f1}"
        )

        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(), f"{model_destination}/best_{model_name}_model.pt"
            )
            print("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping")
            break

    # Save model ở trạng thái cuối cùng
    torch.save(model.state_dict(), "last_model.pth")
    print("Model saved.")
    model.save(folder_path=model_destination, file_name=model_name, type=".pt")
    print("Model saved.")

    # After the training loop and any early stopping logic
    history_file_path = f"{model_destination}/{model_name}_history.json"
    with open(history_file_path, "w") as history_file:
        json.dump(history, history_file)

    print(f"Training history saved to {history_file_path}")
    return model
