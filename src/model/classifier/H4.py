import torch
import torch.nn as nn
from torchvision.models import vit_b_16  # Sử dụng mô hình Vision Transformer B-16
from src.model.classifier.H0 import H0


class H4(H0):
    def __init__(self):
        super(H4, self).__init__()
        # Tải mô hình Vision Transformer B-16 đã được huấn luyện trước
        vit = vit_b_16(pretrained=True)
        # Loại bỏ lớp classification head
        self.feature_extractor = nn.Sequential(*list(vit.children())[:-1])
        # Đóng băng các tham số trong feature extractor để không cập nhật trong quá trình huấn luyện
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Kích thước đầu vào cho lớp fully connected đầu tiên dựa trên output của ViT-B/16
        # ViT-B/16 thường trả về tensor [batch_size, num_patches + 1, dim] sau transformer
        # Lấy dim từ 'dim' của ViT-B/16
        dim = vit.heads[0].in_features
        self.fc1 = nn.Linear(dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        # Output layer với 3 units cho 3 lớp phân loại
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # Trích xuất đặc trưng
        x = self.feature_extractor(x)
        
        # Flatten the spatial dimensions before passing to ViT
        x = x.flatten(2).transpose(1, 2)  # Reshape to (batch_size, seq_length, hidden_dim)
        
        # Lấy token lớp [CLS] làm đại diện đặc trưng cho toàn bộ ảnh
        # Token [CLS] là phần tử đầu tiên trong sequence output của ViT
        x = x[:, 0]  # Extract the [CLS] token
        x = self.classifier(x)
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
            # Lấy token [CLS] làm đặc trưng
            cls_token = feature_maps[:, 0]

        return (cls_token,)