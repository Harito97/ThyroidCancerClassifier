import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from src.model.classifier.H0 import H0
from src.data_preparation.ThyroidCancerDataset import ThyroidCancerDataset
from src.data_preparation.ThyroidCancerDataLoader import ThyroidCancerDataLoader

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  # Lấy toàn bộ trừ lớp FC và avgpool
    
    def forward(self, x):
        x = self.features(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_dim=2048, dim=256, num_classes=3):
        super(VisionTransformer, self).__init__()
        self.dim = dim

        self.flatten = nn.Flatten(2)
        self.linear_proj = nn.Linear(input_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional_encodings = nn.Parameter(torch.zeros(50, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (B, C, H, W) -> (B, C, H*W) -> (H*W, B, C)
        x = self.linear_proj(x)  # Linear projection (H*W, B, dim)
        
        cls_tokens = self.cls_token.expand(-1, B, -1)  # (1, B, dim)
        x = torch.cat((cls_tokens, x), dim=0)  # (1 + H*W, B, dim)
        
        # Cập nhật giá trị của positional_encodings
        self.positional_encodings.data = self.positional_encodings.data[0:x.size(0)]  # Cắt positional_encodings nếu cần thiết
        x += self.positional_encodings

        x = self.transformer(x)  # (1 + H*W, B, dim)
        x = self.fc(x[0])  # Lấy token cls và đưa qua FC layer
        x = self.output(x)  # (B, num_classes)
        return x


class H9(H0):
    def __init__(self, num_classes=3):
        super(H9, self).__init__()
        self.num_classes = num_classes
        self.resnet50_feature_extractor = ResNet50FeatureExtractor()
        self.vit = VisionTransformer(input_dim=2048, dim=256, num_classes=num_classes)
        self.model = nn.Sequential(self.resnet50_feature_extractor, self.vit)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def get_feature_maps(self, x):
        """
        Sẽ chỉ trả về bản đồ đặc trưng sau khi qua ResNet50
        """
        x = self.resnet50_feature_extractor(x)
        return x

    def load_data(self, data_dir, classes={0: ["B2"], 1: ["B5"], 2: ["B6"]}, balance=False):
        print('Creating dataset...')
        train_dataset = ThyroidCancerDataset(img_dir=data_dir, transform=None, classes=classes, balance=balance, mode='train')
        print('Train dataset size:', train_dataset.__len__())
        valid_dataset = ThyroidCancerDataset(img_dir=data_dir, transform=None, classes=classes, balance=False, mode='valid')
        print('Valid dataset size:', valid_dataset.__len__())
        test_dataset = ThyroidCancerDataset(img_dir=data_dir, transform=None, classes=classes, balance=False, mode='test')
        print('Test dataset size:', test_dataset.__len__())

        print('Creating dataloader...')
        thyroidCancerDataLoader = ThyroidCancerDataLoader()
        self.train_loader = thyroidCancerDataLoader.get_data_loader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        print('Train loader size:', len(self.train_loader))
        self.valid_loader = thyroidCancerDataLoader.get_data_loader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
        print('Valid loader size:', len(self.valid_loader))
        self.test_loader = thyroidCancerDataLoader.get_data_loader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        print('Test loader size:', len(self.test_loader))

    def __setup_hyperparameters(
        self,
        train_dataset=None,
        class_weights=None,
    ):
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

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        class_weights_tensor = class_weights_tensor.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def __train(
        self,
        num_epoch=100,
        patience=30,
        model_destination=".",
        model_name="model",
    ):
        if self.train_loader is None or self.valid_loader is None:
            print("No data loader is provided")
            return
        if self.model is None:
            print("No model is provided")
            return
        if self.criterion is None or self.optimizer is None:
            print("No criterion or optimizer is provided")
            return

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        print("Training classification model...")
        best_loss = float("inf")
        patience_counter = 0

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        model_destination = (
            model_destination[:-1] if model_destination[-1] == "/" else model_destination
        )
        model_name = model_name.split(".")[0]
        history_file_path = f"{model_destination}/{model_name}_history.json"

        for epoch in range(num_epoch):
            self.model.train()
            running_loss = 0.0
            train_preds, train_targets = []
            print(f"Epoch {epoch+1}/{num_epoch}:\nStart with batch size: ", end="")
            for images, labels in self.train_loader:
                print(
                    f"[{images.size(0)}, {images.size(1)}, {images.size(2)}, {images.size(3)}]",
                    end=" ",
                )
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.view(-1).cpu().numpy())
                train_targets.extend(labels.view(-1).cpu().numpy())

            train_loss = running_loss / len(self.train_loader)
            train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
            train_f1 = f1_score(train_targets, train_preds, average="weighted")

            self.model.eval()
            print(f"\nStart validation at epoch {epoch + 1} ...")
            val_running_loss = 0.0
            val_preds, val_targets = []
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

            print(
                f"Epoch [{epoch + 1}/{num_epoch}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            with open(history_file_path, "w") as f:
                json.dump(history, f)
                print(f"History saved at {history_file_path}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{model_destination}/{model_name}.pth")
                print(f"Best model saved at {model_destination}/best_{model_name}.pth")
            else:
                patience_counter += 1
                torch.save(self.model.state_dict(), f"{model_destination}/{model_name}.pth")
                print(f"Last model saved at {model_destination}/last_{model_name}.pth")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


    def fit(
        self,
        num_epoch=100,
        patience=30,
        model_destination=".",
        model_name="model",
    ):
        self.__setup_hyperparameters(train_dataset=self.train_loader.dataset)
        self.__train(num_epoch=num_epoch, patience=patience, model_destination=model_destination, model_name=model_name)

    def test(
        self,
        model_path=None,
        show_cm=True,
        show_cr=True,
        show_auc=False,
    ):
        if self.test_loader is None:
            print("No test loader is provided")
            return
        if self.model is None:
            print("No model is provided")
            return
        if self.criterion is None:
            print("No criterion is provided")
            return

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_targets.extend(labels.view(-1).cpu().numpy())

        test_loss /= len(self.test_loader)
        test_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        test_f1 = f1_score(all_targets, all_preds, average="weighted")

        print(
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}"
        )

        if show_cm:
            cm = confusion_matrix(all_targets, all_preds)
            df_cm = pd.DataFrame(cm, index=range(self.num_classes), columns=range(self.num_classes))
            plt.figure(figsize=(10, 7))
            sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()

        if show_cr:
            print(classification_report(all_targets, all_preds, digits=4))

        if show_auc:
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(
                    np.array(all_targets) == i, np.array(all_preds) == i
                )
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure()
            for i in range(self.num_classes):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
                )
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            plt.show()
        all_targets, all_preds = np.array(all_targets), np.array(all_preds)
        return all_targets, all_preds
