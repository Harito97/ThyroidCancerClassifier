import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from src.model.classifier.H0 import H0


class H8_0_12(H0):
    """
    Dùng để phân biệt nhãn 0 và nhãn 1 và 2 
    """
    def __init__(self, num_classes=2):
        super(H5, self).__init__()
        
        # Tải mô hình ResNet50 đã được huấn luyện trước
        resnet50 = models.resnet50(pretrained=True)
        # Loại bỏ lớp fully connected cuối cùng
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])
        # Đóng băng các tham số trong feature extractor để không cập nhật trong quá trình huấn luyện
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Tải mô hình ViT đã được huấn luyện trước
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        # Đóng băng các tham số của ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Linear layer để chuyển đổi đặc trưng từ ResNet sang định dạng đầu vào của ViT
        self.resnet_to_vit = nn.Linear(2048, self.vit.embed_dim)

        # Thay thế đầu ra của ViT bằng một lớp fully connected
        self.vit.head = nn.Linear(self.vit.head.in_features, 512)

        # Layer fully connected cuối cùng cho 2 lớp phân loại
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Trích xuất đặc trưng từ ResNet50
        x = self.feature_extractor(x)
        # Chuyển đổi kích thước tensor từ [batch_size, 2048, 7, 7] sang [batch_size, 2048]
        x = torch.flatten(x, 2)
        # Chuyển đổi kích thước tensor từ [batch_size, 2048, 49] sang [batch_size, 49, 2048]
        x = x.permute(0, 2, 1)
        # Chuyển đổi đặc trưng từ ResNet sang định dạng đầu vào của ViT
        x = self.resnet_to_vit(x)

        # Đưa qua ViT
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = self.vit.head(x[:, 0])

        # Đưa qua lớp fully connected cuối cùng
        x = self.fc(x)

        return x

    def get_feature_maps(self, x):
        """
        Trả về bản đồ đặc trưng cho đầu vào x
        :param x: Tensor đầu vào có kích thước (N, C, H, W)
        :return: Bản đồ đặc trưng
        """
        # Trích xuất đặc trưng từ ResNet50
        x = self.feature_extractor(x)
        # Chuyển đổi kích thước tensor từ [batch_size, 2048, 7, 7] sang [batch_size, 2048]
        x = torch.flatten(x, 2)
        # Chuyển đổi kích thước tensor từ [batch_size, 2048, 49] sang [batch_size, 49, 2048]
        x = x.permute(0, 2, 1)
        # Chuyển đổi đặc trưng từ ResNet sang định dạng đầu vào của ViT
        x = self.resnet_to_vit(x)

        # Đưa qua ViT
        feature_maps = self.vit.blocks(x)
        feature_maps = self.vit.norm(feature_maps)

        return feature_maps
