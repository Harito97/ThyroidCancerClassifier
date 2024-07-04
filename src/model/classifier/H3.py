import torch
import torch.nn as nn
from torchvision import models
from src.model.classifier.H0 import H0


class H3(H0):
    def __init__(self):
        super(H3, self).__init__()
        # Tải mô hình ResNet50 đã được huấn luyện trước
        resnet50 = models.resnet50(pretrained=True)
        # Loại bỏ lớp fully connected cuối cùng
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        # Đóng băng các tham số trong feature extractor để không cập nhật trong quá trình huấn luyện
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Kích thước đầu vào cho lớp fully connected đầu tiên dựa trên output của ResNet50
        # ResNet50 thường trả về tensor [batch_size, 2048, 1, 1] sau lớp pooling cuối cùng
        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(0.5)
        # Output layer với 3 units cho 3 lớp phân loại
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # Trích xuất đặc trưng
        x = self.feature_extractor(x)
        # Chuyển đổi tensor từ [batch_size, 2048, 1, 1] sang [batch_size, 2048] để phù hợp với lớp fully connected
        x = torch.flatten(x, 1)
        # Đưa qua mạng dense
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def get_feature_maps(self, x):
        """
        Trả về bản đồ đặc trưng cho đầu vào x
        :param x: Tensor đầu vào có kích thước (N, C, H, W)
        :return: Bản đồ đặc trưng
        """
        # Đảm bảo mô hình ở chế độ đánh giá
        self.feature_extractor.eval()

        # Tính toán bản đồ đặc trưng
        with torch.no_grad():
            feature_maps = self.feature_extractor(x)

        return (feature_maps,)
