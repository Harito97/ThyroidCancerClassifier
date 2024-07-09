import os
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import wandb


class Tool:
    @staticmethod
    def save_confusion_matrix(y_true, y_score, target_names, filename, normalize=False):
        """
        Saves the confusion matrix to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            target_names (list): Names of the target classes.
            filename (str): Path to save the confusion matrix.
            normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
        Returns:
            cm (numpy.ndarray): The confusion matrix.
        """
        try:
            cm = confusion_matrix(y_true, y_score)
            print("Confusion Matrix:\n", cm)
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
            return cm
        except ValueError as e:
            print(f"Error creating confusion matrix: {e}")
            return None

    @staticmethod
    def save_classification_report(y_true, y_score, filename):
        """
        Saves the classification report to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            filename (str): Path to save the classification report.
        Returns:
            cr (dict): The classification report.
        """
        try:
            cr = classification_report(y_true, y_score, output_dict=True)
            print("Classification Report:\n", cr)
            report_df = pd.DataFrame(cr).transpose()
            # report_df.drop("support", axis=1, inplace=True)  # Bỏ cột support nếu không cần
            report_df.plot(kind="bar", figsize=(10, 6))
            plt.title("Classification Report")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return cr
        except ValueError as e:
            print(f"Error creating DataFrame from classification report: {e}")
            return None

    @staticmethod
    def save_roc_auc_plot(y_true, y_score, n_classes, filename):
        """
        Calculates and saves the ROC AUC plot to a file.

        Args:
            y_true (array-like): True labels.
            y_score (array-like): Predicted scores.
            n_classes (int): Number of classes.
            filename (str): Path to save the plot.
        Returns:
            fpr (dict): False positive rates for each class.
            tpr (dict): True positive rates for each class.
            roc_auc (dict): ROC AUC scores for each class.
        """
        try:
            # Binarize the output
            if n_classes != 2:
                y_true = label_binarize(y_true, classes=[*range(n_classes)])

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure(figsize=(8, 6))

            if n_classes == 2:
                plt.plot(
                    fpr[1],
                    tpr[1],
                    lw=2,
                    label="ROC curve (area = {0:0.2f})".format(roc_auc[1]),
                )
            else:
                for i in range(n_classes):
                    plt.plot(
                        fpr[i],
                        tpr[i],
                        lw=2,
                        label="ROC curve of class {0} (area = {1:0.2f})".format(
                            i, roc_auc[i]
                        ),
                    )

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            plt.savefig(filename)
            plt.close()
            return fpr, tpr, roc_auc
        except ValueError as e:
            print(f"Error creating ROC AUC plot: {e}")
            return None, None, None


def test(
    test_loader=None,
    model=None,  # model structure
    device=None,
    criterion=None,
    model_destination=".",
    model_name="model",
):
    """
    Tests a given model on provided data and saves the results.

    Args:
        test_loader (DataLoader, optional): DataLoader for the test data. Defaults to None.
        model (nn.Module, optional): The model to test. Defaults to None.
        device (torch.device, optional): The device to run the test on. Defaults to None.
        criterion (nn.Module, optional): The loss function. Defaults to None.
        model_destination (str, optional): The directory to save the results in. Defaults to ".".
        model_name (str, optional): The name of the model. Defaults to "model".

    Returns:
        None
    """

    if test_loader is None:
        print("No data loader is provided")
        return
    if model is None or device is None:
        print("No model or device is provided")
        return
    if criterion is None:
        print("No criterion is provided")
        return

    print("Loading model weights...")
    model_destination = (
        model_destination[:-1] if model_destination[-1] == "/" else model_destination
    )
    model_name = model_name.split(".")[0]
    model_path = f"{model_destination}/best_{model_name}_model.pt"
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        return
    model = model.to(device)
    criterion = criterion.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluate mode

    # Initialize W&B
    wandb.init(project="ThyroidCancer", entity="harito97")

    # Test loop
    test_preds, test_targets, test_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        total_batches = len(test_loader)  # Get total number of batches

        for i, (images, labels) in enumerate(test_loader):
            progress = (i + 1) / total_batches * 100  # Calculate progress
            print(f"\rProgress: {progress:.2f}%", end="")

            # Load data of 1 batch to device
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.view(-1).cpu().numpy())
            test_targets.extend(labels.view(-1).cpu().numpy())
            test_probs.extend(outputs.cpu().numpy())  # Save probabilities

    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    test_f1 = f1_score(test_targets, test_preds, average="weighted")

    print(
        f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}, Test F1: {test_f1:.6f}"
    )

    # Log metrics to W&B
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})

    # Confusion Matrix and Classification Report
    unique_labels = np.unique(test_targets)
    target_names = [str(label) for label in unique_labels]
    cm = Tool.save_confusion_matrix(
        y_true=test_targets,
        y_score=test_preds,
        target_names=target_names,
        filename=f"{model_destination}/confusion_matrix.png",
    )
    cr = Tool.save_classification_report(
        y_true=test_targets,
        y_score=test_preds,
        filename=f"{model_destination}/classification_report.png",
    )
    fpr, tpr, roc_auc = Tool.save_roc_auc_plot(
        y_true=test_targets,
        y_score=test_probs,
        n_classes=len(unique_labels),
        filename=f"{model_destination}/roc_auc_plot.png",
    )

    # Log additional metrics and plots to W&B
    wandb.log(
        {
            "confusion_matrix": wandb.Image(
                f"{model_destination}/confusion_matrix.png"
            ),
            "classification_report": wandb.Image(
                f"{model_destination}/classification_report.png"
            ),
            "roc_auc_plot": wandb.Image(f"{model_destination}/roc_auc_plot.png"),
        }
    )

    # Save test information to npz file
    # model_name = os.path.basename(model_path).split(".")[0]
    # result_destination = os.path.dirname(model_path)
    print(
        f"Saving test metrics to {os.path.join(model_destination, f'test_{model_name}_metrics.npz')}"
    )
    np.savez(
        os.path.join(model_destination, f"test_{model_name}_metrics.npz"),
        test_preds=test_preds,
        test_probs=test_probs,
        test_targets=test_targets,
        test_loss=test_loss,
        test_acc=test_acc,
        test_f1=test_f1,
        target_names=target_names,
        cm=cm,
        cr=cr,
        roc_auc=roc_auc,
        fpr=fpr,
        tpr=tpr,
    )
    print("Test completed")
    wandb.finish()
