import torch
import torch.nn as nn
import torchvision.models as models
# from torchsummary import summary

class H97_ResNet(nn.Module):
    def __init__(self, num_classes: int = 3, retrainResNet: bool = False):
        super(H97_ResNet, self).__init__()        
        # Tải mô hình ResNet50 đã được huấn luyện trước
        resnet50 = models.resnet50(pretrained=True)
        # Loại bỏ lớp fully connected cuối cùng
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        if not retrainResNet:
            # Đóng băng các tham số trong feature extractor để không cập nhật trong quá trình huấn luyện
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Kích thước đầu vào cho lớp fully connected đầu tiên dựa trên output của ResNet50
        # ResNet50 thường trả về tensor [batch_size, 2048, 1, 1] sau lớp pooling cuối cùng
        self.fc1 = nn.Linear(2048, 9)
        self.fc2 = nn.Linear(9, 7)
        self.fc3 = nn.Linear(7, 3)
        self.fc4 = nn.Linear(3, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Trích xuất đặc trưng
        x = self.feature_extractor(x)
        # Chuyển đổi tensor từ [batch_size, 2048, 1, 1] sang [batch_size, 2048] để phù hợp với lớp fully connected
        x = torch.flatten(x, 1)
        # Đưa qua mạng dense
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

# # Khởi tạo mô hình
# model = H97_ResNet()

# # Trực quan hóa kiến trúc mô hình
# summary(model, (3, 224, 224))  # Input shape (3 channels, 224x224 image)

class H97_EfficientNet(nn.Module):
    def __init__(self, num_classes: int = 3, retrainEfficientNet: bool = False):
        super(H97_EfficientNet, self).__init__()
        # Load a pretrained EfficientNet model
        efficientnet = models.efficientnet_b0(pretrained=True)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        if not retrainEfficientNet:
            # Freeze the parameters in the feature extractor to not update during training
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # The input size for the first fully connected layer based on the output of EfficientNet
        # EfficientNet typically returns a tensor [batch_size, 1280, 1, 1] after the last pooling layer
        self.fc1 = nn.Linear(1280, 9)
        self.fc2 = nn.Linear(9, 7)
        self.fc3 = nn.Linear(7, 3)
        self.fc4 = nn.Linear(3, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        # Flatten the tensor from [batch_size, 1280, 1, 1] to [batch_size, 1280] to match the fully connected layer
        x = torch.flatten(x, 1)
        # Pass through the dense network
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x