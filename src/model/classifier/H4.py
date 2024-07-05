import timm
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.model.classifier.H0 import H0


class H4(H0):
    def __init__(self, num_classes=3):
        super(H4, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Freeze all layers except the head
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

    def get_all_feature_maps(self, x):
        """
        Trả về toàn bộ bản đồ đặc trưng cho đầu vào x
        :param x: Tensor đầu vào có kích thước (N, C, H, W)
        :return: Toàn bộ bản đồ đặc trưng
        """
        return None