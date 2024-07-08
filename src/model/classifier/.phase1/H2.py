import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.classifier.H0 import H0
from src.model.classifier.H1 import H1

class H2(H0):
    def __init__(self):
        super(H2, self).__init__()

        #######################################################

        # Khởi tạo trọng số cho lớp Convolution để làm nổi bật các góc cạnh
        # Khởi tạo bộ lọc Sobel dọc
        sobel_filter_vertical = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32
        ).repeat(3, 3, 1, 1)  # Lặp lại cho mỗi kênh màu

        # Khởi tạo bộ lọc Sobel ngang (tuỳ chọn, tùy thuộc vào yêu cầu)
        sobel_filter_horizontal = torch.tensor(
            [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]],
            dtype=torch.float32
        ).repeat(3, 3, 1, 1)  # Lặp lại cho mỗi kênh màu

        # Kết hợp bộ lọc dọc và ngang (tuỳ chọn)
        sobel_filter = sobel_filter_vertical + sobel_filter_horizontal

        ## Khởi tạo lớp Conv2d
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        ## Đặt trọng số cho lớp Conv2d
        self.conv0.weight.data = sobel_filter
        ## Lớp Convolution đầu tiên không được train
        self.conv0.weight.requires_grad = False

        #######################################################

        # Các lớp của mô hình H1
        self.H1 = H1()

    def forward(self, x):
        x = self.conv0(x)
        x = self.H1(x)
        return x

    def get_feature_maps(self, x):
        # Phương thức mới để trả về bản đồ đặc trưng
        features0 = F.relu(self.conv0(x))
        features1, features2, features3 = self.H1.get_feature_maps(x)
        # Trả về bản đồ đặc trưng từ mỗi lớp convolutional
        return features0, features1, features2, features3
