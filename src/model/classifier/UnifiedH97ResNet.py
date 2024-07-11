from torch import cat, load
from torch.nn import Linear, Module, ReLU, Sequential, Dropout, Softmax
from torchvision.models import resnet50

# from src.model.classifier.H0 import H0
"""
Sau đây là script khởi tạo UnifiedModel 
(load trọng số của 4 model đã huấn luyện trước vào 4 lớp fully connected của UnifiedModel) 
và script khởi tạo mô hình UnifiedH97ResNet 
(kết hợp mô hình ResNet50 và UnifiedModel để phân loại 3 nhãn B2, B5, B6 từ 9 đặc trưng)
B1. Load trọng số của 4 model đã huấn luyện trước vào 4 lớp fully connected của UnifiedModel

"""

class UnifiedModel(Module):
    def __init__(self, weights_path=None):
        super(UnifiedModel, self).__init__()
        # Định nghĩa các lớp fully connected cho mỗi model
        self.fc1_B2_B5B6 = Linear(2048, 9)
        self.fc2_B2_B5B6 = Linear(9, 7)
        self.fc3_B2_B5B6 = Linear(7, 3)
        self.fc4_B2_B5B6 = Linear(3, 2)

        self.fc1_B5_B2B6 = Linear(2048, 9)
        self.fc2_B5_B2B6 = Linear(9, 7)
        self.fc3_B5_B2B6 = Linear(7, 3)
        self.fc4_B5_B2B6 = Linear(3, 2)

        self.fc1_B6_B2B5 = Linear(2048, 9)
        self.fc2_B6_B2B5 = Linear(9, 7)
        self.fc3_B6_B2B5 = Linear(7, 3)
        self.fc4_B6_B2B5 = Linear(3, 2)

        self.fc1_B2_B5_B6 = Linear(2048, 9)
        self.fc2_B2_B5_B6 = Linear(9, 7)
        self.fc3_B2_B5_B6 = Linear(7, 3)
        self.fc4_B2_B5_B6 = Linear(3, 3)

        # Dropout layer
        self.dropout = Dropout(0.1)

        # Tự động tải trọng số nếu đường dẫn được cung cấp
        if weights_path is not None:
            self.load_state_dict(load(weights_path))

    def forward(self, x):
        # Đưa qua mạng dense của 4 model con
        x_B2_B5B6 = self.__forward_B2_B5B6(x)
        x_B5_B2B6 = self.__forward_B5_B2B6(x)
        x_B6_B2B5 = self.__forward_B6_B2B5(x)
        x_B2_B5_B6 = self.__forward_B2_B5_B6(x)

        # Kết hợp output từ 4 model con
        x = cat((x_B2_B5B6, x_B5_B2B6, x_B6_B2B5, x_B2_B5_B6), dim=1)
        return x

    def __forward_B2_B5B6(self, x):
        x_B2_B5B6 = self.fc1_B2_B5B6(x)
        x_B2_B5B6 = ReLU()(x_B2_B5B6)
        x_B2_B5B6 = self.dropout(x_B2_B5B6)
        x_B2_B5B6 = self.fc2_B2_B5B6(x_B2_B5B6)
        x_B2_B5B6 = ReLU()(x_B2_B5B6)
        x_B2_B5B6 = self.dropout(x_B2_B5B6)
        x_B2_B5B6 = self.fc3_B2_B5B6(x_B2_B5B6)
        x_B2_B5B6 = ReLU()(x_B2_B5B6)
        x_B2_B5B6 = self.dropout(x_B2_B5B6)
        x_B2_B5B6 = self.fc4_B2_B5B6(x_B2_B5B6)
        return x_B2_B5B6

    def __forward_B5_B2B6(self, x):
        x_B5_B2B6 = self.fc1_B5_B2B6(x)
        x_B5_B2B6 = ReLU()(x_B5_B2B6)
        x_B5_B2B6 = self.dropout(x_B5_B2B6)
        x_B5_B2B6 = self.fc2_B5_B2B6(x_B5_B2B6)
        x_B5_B2B6 = ReLU()(x_B5_B2B6)
        x_B5_B2B6 = self.dropout(x_B5_B2B6)
        x_B5_B2B6 = self.fc3_B5_B2B6(x_B5_B2B6)
        x_B5_B2B6 = ReLU()(x_B5_B2B6)
        x_B5_B2B6 = self.dropout(x_B5_B2B6)
        x_B5_B2B6 = self.fc4_B5_B2B6(x_B5_B2B6)
        return x_B5_B2B6

    def __forward_B6_B2B5(self, x):
        x_B6_B2B5 = self.fc1_B6_B2B5(x)
        x_B6_B2B5 = ReLU()(x_B6_B2B5)
        x_B6_B2B5 = self.dropout(x_B6_B2B5)
        x_B6_B2B5 = self.fc2_B6_B2B5(x_B6_B2B5)
        x_B6_B2B5 = ReLU()(x_B6_B2B5)
        x_B6_B2B5 = self.dropout(x_B6_B2B5)
        x_B6_B2B5 = self.fc3_B6_B2B5(x_B6_B2B5)
        x_B6_B2B5 = ReLU()(x_B6_B2B5)
        x_B6_B2B5 = self.dropout(x_B6_B2B5)
        x_B6_B2B5 = self.fc4_B6_B2B5(x_B6_B2B5)
        return x_B6_B2B5

    def __forward_B2_B5_B6(self, x):
        x_B2_B5_B6 = self.fc1_B2_B5_B6(x)
        x_B2_B5_B6 = ReLU()(x_B2_B5_B6)
        x_B2_B5_B6 = self.dropout(x_B2_B5_B6)
        x_B2_B5_B6 = self.fc2_B2_B5_B6(x_B2_B5_B6)
        x_B2_B5_B6 = ReLU()(x_B2_B5_B6)
        x_B2_B5_B6 = self.dropout(x_B2_B5_B6)
        x_B2_B5_B6 = self.fc3_B2_B5_B6(x_B2_B5_B6)
        x_B2_B5_B6 = ReLU()(x_B2_B5_B6)
        x_B2_B5_B6 = self.dropout(x_B2_B5_B6)
        x_B2_B5_B6 = self.fc4_B2_B5_B6(x_B2_B5_B6)
        return x_B2_B5_B6


class UnifiedH97ResNet(Module):
    def __init__(self, unified_model_weights_path=None):
        super(UnifiedH97ResNet, self).__init__()

        # Module ResNet
        resnet50_pretrained = resnet50(pretrained=True)
        self.feature_extractor = Sequential(*list(resnet50_pretrained.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Module UnifiedModel
        self.unified_model = UnifiedModel(unified_model_weights_path)
        for param in self.unified_model.parameters():
            param.requires_grad = False

        # Dropout layer
        self.dropout = Dropout(0.1)

        # Lớp cuối cùng để phân loại 3 nhãn B2, B5, B6 từ 9 đặc trưng
        self.near_last_fc = Linear(9, 9)
        self.final_fc = Linear(9, 3)

    def forward(self, x):
        # Trích xuất đặc trưng qua module ResNet
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # Chuyển đổi tensor từ [batch_size, 2048, 1, 1] sang [batch_size, 2048]

        # Đưa qua mạng dense của UnifiedModel
        x = self.unified_model(x)
        x = ReLU()(x)
        x = self.dropout(x)

        # Cho vào những lớp cuối cùng để phân loại 3 nhãn B2, B5, B6
        x = self.near_last_fc(x)
        x = ReLU()(x)
        x = self.dropout(x)
        x = self.final_fc(x)
        x = Softmax(dim=1)(x)
        return x
