import os
import json
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize



def __load_data(data_version_dir, for_training=True):
    print("Loading data...")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                # Đây là các giá trị đã tính trên tập train
                mean=[0.66741932, 0.59166461, 0.82794493],
                std=[0.25135074, 0.26329945, 0.11295287],
            ),
        ]
    )

    if not for_training:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    # Đây là các giá trị đã tính trên tập train
                    mean=[0.66741932, 0.59166461, 0.82794493],
                    std=[0.25135074, 0.26329945, 0.11295287],
                ),
            ]
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=f"{data_version_dir}/test", transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return test_loader

    train_dataset = torchvision.datasets.ImageFolder(
        root=f"{data_version_dir}/train", transform=transform
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        root=f"{data_version_dir}/valid", transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    return train_loader, valid_loader


def __prepare_model(model):
    if model is None:
        print("No model is provided")
        return
    print("Setting up model to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def __setup_hyperparameters(
    model, train_dataset=None, class_weights=[4.6748, 0.8772, 0.6075], device="cpu"
):
    # class_weights=[4.6748, 0.8772, 0.6075] là trọng số của mỗi class trong hàm loss
    # đã tính dựa trên phân phối data tập train
    print("Setting up loss function and optimizer...")
    if class_weights is None:
        if train_dataset is None:
            print("No class weights as no train dataset is provided")
            return
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_dataset.targets),
            y=train_dataset.targets,
        )

    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    # tensor([4.6748, 0.8772, 0.6075])

    # Move to the correct device
    class_weights_tensor = class_weights_tensor.to(device)

    # Use the weights in the loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    if train_dataset is None:
        print("Return only criterion with no optimizer")
        return criterion

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, optimizer


def __train(
    train_loader=None,
    valid_loader=None,
    model=None,
    device=None,
    criterion=None,
    optimizer=None,
    num_epoch=100,
    patience=30,
    model_destination=".",
    model_name="model",
):
    """
    model_destination: str không được có dấu / ở cuối
    model_name: str không có dấu . trong tên
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

    model = model.to(device)
    criterion = criterion.to(device)
    # optimizer = optimizer.to(device)

    print("Training classification model...")
    best_loss = float("inf")
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
        # Đặt mô hình ở chế độ train
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        print(f"Epoch {epoch+1}/{num_epoch}:\nStart with batch size: ", end="")
        for images, labels in train_loader:
            print(images.size(0), end=" ")
            # Load vào dữ liệu 1 batch
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

        # Calculate metrics sau epoch này trên tập train
        train_loss = running_loss / len(train_loader)
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_f1 = f1_score(train_targets, train_preds, average="weighted")

        # Validation loop
        # Đặt mô hình ở chế độ kiểm thử
        model.eval()
        print(f"\nStart validation at epoch {epoch + 1} ...")
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

        print(
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Train F1: {train_f1:.6f}\n"
            f"Val   Loss: {val_loss:.6f}, Val   Acc: {val_acc:.6f}, Val   F1: {val_f1:.6f}"
        )

        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(), f"{model_destination}/best_{model_name}_model.pt"
            )
            print("Saved best model at epoch", epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping")
            break

    # Save model ở trạng thái cuối cùng
    model_destination = (
        model_destination[:-1] if model_destination[-1] == "/" else model_destination
    )
    model_name = model_name.split(".")[0]

    torch.save(model.state_dict(), f"{model_destination}/last_{model_name}_model.pt")
    print("Saved last model")

    # After the training loop and any early stopping logic
    history_file_path = f"{model_destination}/{model_name}_history.json"
    with open(history_file_path, "w") as history_file:
        json.dump(history, history_file)
    print(f"Training history saved to {history_file_path}")
    print("Training completed")


def fit(
    model=None,
    data_version_dir=None,
    num_epoch: int = 100,
    model_destination: str = ".",
    model_name: str = "model",
    mean=[0.66741932, 0.59166461, 0.82794493],
    std=[0.25135074, 0.26329945, 0.11295287],
):
    # Step 1. Load data and transform data
    train_loader, valid_loader = __load_data(data_version_dir)

    # Step 2. Prepare model to device
    model, device = __prepare_model(model)

    # Step 3. Setup hyperparameters: Prepare loss function and optimizer
    criterion, optimizer = __setup_hyperparameters(model, train_loader.dataset)

    # Step 4. Training loop
    __train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epoch=num_epoch,
        model_destination=model_destination,
        model_name=model_name,
    )


# Lưu Confusion Matrix
def __save_confusion_matrix(cm, target_names, filename, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix, Without Normalization"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# Lưu Classification Report
def __save_classification_report(cr, filename):
    try:
        report_df = pd.DataFrame(cr).transpose()
        report_df.drop("support", axis=1, inplace=True)  # Bỏ cột support nếu không cần
        report_df.plot(kind="bar", figsize=(10, 6))
        plt.title("Classification Report")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except ValueError as e:
        print(f"Error creating DataFrame from classification report: {e}")


# Lưu ROC AUC Plot
def __save_roc_auc_plot(fpr, tpr, roc_auc, n_classes, filename):
    """
    Saves the ROC AUC plot to a file.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        roc_auc: ROC AUC scores.
        n_classes: Number of classes.
        filename: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))

    # Vẽ đường ROC cho từng lớp
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    # Vẽ đường chéo (ngẫu nhiên)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    # Thiết lập các thông số của đồ thị
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    # Lưu đồ thị vào file
    plt.savefig(filename)
    plt.close()


def test(
    model=None,  # model structure
    data_version_dir=None,
    model_path=None,
    criterion=None,
    mean=None,
    std=None,
):
    if mean is None:
        mean = [0.66741932, 0.59166461, 0.82794493]
    if std is None:
        std = [0.25135074, 0.26329945, 0.11295287]

    # Load data
    print("Loading test data...")
    test_loader = __load_data(data_version_dir, for_training=False)

    # Prepare model
    print("Preparing model...")
    model, device = __prepare_model(model)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Test loop
    test_preds, test_targets, test_probs = [], [], []
    total_loss = 0
    print(f"Start test with batch size: ", end="")
    model.eval()  # Chuyển model sang chế độ đánh giá
    with torch.no_grad():
        for images, labels in test_loader:
            print(images.size(0), end=" ")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.view(-1).cpu().numpy())
            test_targets.extend(labels.view(-1).cpu().numpy())
            test_probs.extend(outputs.cpu().numpy())  # Lưu giá trị xác suất

    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    test_f1 = f1_score(test_targets, test_preds, average="weighted")

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(test_targets, test_preds)
    cr = classification_report(test_targets, test_preds)
    cr_dict = classification_report(test_targets, test_preds, output_dict=True)

    print(
        f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}, Test F1: {test_f1:.6f}"
    )
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # Tính ROC, AUC cho mỗi nhãn
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    n_classes = test_probs.shape[1]
    test_targets_binarized = label_binarize(test_targets, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_targets_binarized[:, i], test_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Save test_preds, test_probs, and test_targets to npz file
    model_name = os.path.basename(model_path).split(".")[0]
    result_destination = os.path.dirname(model_path)
    print(
        f"Saving test metrics to {os.path.join(result_destination, f'test_{model_name}_metrics.npz')}"
    )
    np.savez(
        os.path.join(result_destination, f"test_{model_name}_metrics.npz"),
        test_preds=test_preds,
        test_probs=test_probs,
        test_targets=test_targets,
        test_loss=test_loss,
        test_acc=test_acc,
        test_f1=test_f1,
        cm=cm,
        cr=cr,
        roc_auc=roc_auc,
        fpr=fpr,
        tpr=tpr,
    )
    print(
        f"Saving confusion matrix plot, classification report plot, and ROC AUC plot in {result_destination}"
    )
    __save_confusion_matrix(
        cm,
        target_names=["B2", "B5", "B6"],
        filename=os.path.join(
            result_destination, f"test_{model_name}_confusion_matrix.png"
        ),
        normalize=False,
    )
    __save_confusion_matrix(
        cm,
        target_names=["B2", "B5", "B6"],
        filename=os.path.join(
            result_destination, f"test_{model_name}_confusion_matrix_normalized.png"
        ),
        normalize=True,
    )
    __save_classification_report(
        cr_dict,
        os.path.join(
            result_destination, f"test_{model_name}_classification_report.png"
        ),
    )
    __save_roc_auc_plot(
        fpr,
        tpr,
        roc_auc,
        n_classes,
        os.path.join(result_destination, f"test_{model_name}_roc_auc_plot.png"),
    )


if __name__ == "__main__":
    from src.model.classifier.H3 import H3

    model = H3()
    # fit(
    #     model=model,
    #     data_version_dir="data/processed/ver1",
    #     num_epoch=100,
    #     model_destination="models",
    #     model_name="h3",
    # )

    criterion = __setup_hyperparameters(model)
    test(
        model=model,
        data_version_dir="data/processed/ver1",
        model_path="models/best_h3_model.pt",
        criterion=criterion,
    )
