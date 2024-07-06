import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from src.model.classifier.H0 import H0

class H7(H0):
    def __init__(self, num_classes=3):
        super(H7, self).__init__()
        # Khởi tạo ViT và ResNet từ timm, bỏ qua lớp dense cuối cùng
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)

        # Kích thước đầu ra của ViT và ResNet
        vit_features = self.vit.num_features
        resnet_features = self.resnet.num_features

        # Lớp tổng hợp và phân loại
        self.classifier = nn.Linear(vit_features + resnet_features, num_classes)

    def forward(self, x):
        # Lấy đặc trưng từ ViT và ResNet
        a = self.vit(x)
        b = self.resnet(x)

        # Tổng hợp đặc trưng
        c = torch.cat((a, b), dim=1)

        # Phân loại
        out = self.classifier(c)
        return out

    def get_feature_maps(self, x):
        """
        Trả về bản đồ đặc trưng cho đầu vào x
        :param x: Tensor đầu vào có kích thước (N, C, H, W)
        :return: Bản đồ đặc trưng
        """
        return None
