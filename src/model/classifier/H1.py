import torch
import torch.nn as nn
import torch.nn.functional as F
from H0 import H0


class H1(H0):
    def __init__(self):
        super(H1, self).__init__()
        # Định nghĩa các tầng CNN
        # Tầng 1: Convolutional layer với 3 input channels (ảnh RGB), 32 output channels, kernel size 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization sau tầng conv1
        # Tầng 2: Convolutional layer với 32 input channels, 64 output channels, kernel size 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization sau tầng conv2
        # Tầng 3: Convolutional layer với 64 input channels, 128 output channels, kernel size 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization sau tầng conv3
        # Pooling layer giảm kích thước bằng cách lấy max trong cửa sổ 2x2
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)  # Dropout để giảm overfitting sau pooling
        # Fully connected layer để phân loại, 128 * 28 * 28 là số đặc trưng đầu vào từ tầng cuối cùng của CNN,
        # giả sử sau 3 lần pooling kích thước ảnh giảm xuống còn 28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout2 = nn.Dropout(0.6)  # Dropout trước tầng output để giảm overfitting
        # Output layer với 3 units cho 3 lớp phân loại
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # Áp dụng các tầng convolutional và non-linearity (ReLU)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)  # Sử dụng lại dropout1
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout1(x)  # Sử dụng lại dropout1
        # Flatten đầu ra trước khi đưa vào tầng fully connected
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def get_feature_maps(self, x):
        # Phương thức mới để trả về bản đồ đặc trưng
        features1 = F.relu(self.conv1(x))
        pooled1 = self.pool(features1)
        features2 = F.relu(self.conv2(pooled1))
        pooled2 = self.pool(features2)
        features3 = F.relu(self.conv3(pooled2))
        pooled3 = self.pool(features3)
        # Trả về bản đồ đặc trưng từ mỗi lớp convolutional
        return features1, features2, features3
