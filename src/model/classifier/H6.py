import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from src.model.classifier.H0 import H0

class H6(H0):
    def __init__(self, num_classes=3):
        super(H6, self).__init__()

        # Tải mô hình ViT đã được huấn luyện trước
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        # Đóng băng các tham số của ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Thay thế đầu ra của ViT bằng một lớp fully connected
        self.vit.head = nn.Identity()  # Sử dụng nn.Identity để loại bỏ đầu ra gốc của ViT

        # Tải mô hình ResNet50 đã được huấn luyện trước
        resnet50 = models.resnet50(pretrained=True)
        # Loại bỏ lớp fully connected cuối cùng
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        # Đóng băng các tham số trong feature extractor để không cập nhật trong quá trình huấn luyện
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Linear layer để chuyển đổi đặc trưng từ ViT sang định dạng đầu vào của ResNet
        self.vit_to_resnet = nn.Linear(self.vit.embed_dim, 224 * 224 * 3)

        # Thêm một lớp ẩn với 512 neuron
        self.hidden_layer = nn.Linear(2048, 512)

        # Layer fully connected cuối cùng cho 3 lớp phân loại
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Đưa qua ViT
        x = self.vit.forward_features(x)
        # Lấy token class (đầu ra đầu tiên) từ ViT
        x = x[:, 0]

        # Chuyển đổi đặc trưng từ ViT sang định dạng đầu vào của ResNet
        x = self.vit_to_resnet(x)
        # Chuyển đổi kích thước tensor từ [batch_size, 224*224*3] sang [batch_size, 3, 224, 224]
        x = x.view(-1, 3, 224, 224)

        # Đưa qua ResNet50
        x = self.feature_extractor(x)

        # Chuyển đổi kích thước tensor từ [batch_size, 2048, 1, 1] sang [batch_size, 2048]
        x = torch.flatten(x, 1)

        # Đưa qua lớp ẩn với 512 neuron
        x = F.relu(self.hidden_layer(x))

        # Đưa qua lớp fully connected cuối cùng
        x = self.fc(x)

        return x

    def get_feature_maps(self, x):
        """
        Trả về bản đồ đặc trưng cho đầu vào x
        :param x: Tensor đầu vào có kích thước (N, C, H, W)
        :return: Bản đồ đặc trưng
        """
        # Đưa qua ViT
        x = self.vit.forward_features(x)
        # Lấy token class (đầu ra đầu tiên) từ ViT
        x = x[:, 0]

        # Chuyển đổi đặc trưng từ ViT sang định dạng đầu vào của ResNet
        x = self.vit_to_resnet(x)
        # Chuyển đổi kích thước tensor từ [batch_size, 224*224*3] sang [batch_size, 3, 224, 224]
        x = x.view(-1, 3, 224, 224)

        # Đưa qua ResNet50
        feature_maps = self.feature_extractor(x)

        return feature_maps
