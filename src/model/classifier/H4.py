import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.model.classifier.H0 import H0


class H4(H0):
    def __init__(self, img_dim=224, patch_dim=16, num_channels=3, num_classes=3, embed_dim=256, num_heads=8, num_encoder_layers=12, dropout_rate=0.2):
        super(H4, self).__init__()
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_patches = (img_dim // patch_dim) ** 2
        self.embed_dim = embed_dim

        self.patch_size = img_dim // patch_dim
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.patch_embedding = nn.Linear(patch_dim * patch_dim * num_channels, embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        # Reshape input to patches
        x = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        x = x.contiguous().view(x.size(0), x.size(1) * x.size(2) * x.size(3), -1)

        # Embed patches
        x = self.patch_embedding(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Classifier
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x


    def get_all_feature_maps(self, x):
        """
        Trả về toàn bộ bản đồ đặc trưng cho đầu vào x
        :param x: Tensor đầu vào có kích thước (N, C, H, W)
        :return: Toàn bộ bản đồ đặc trưng
        """
        # Đảm bảo mô hình ở chế độ đánh giá
        self.feature_extractor.eval()

        # Tính toán bản đồ đặc trưng
        with torch.no_grad():
            feature_maps = self.feature_extractor(x)
            # Trả về toàn bộ bản đồ đặc trưng thay vì chỉ lấy token [CLS]
            # Điều này giúp lấy thông tin đặc trưng đầy đủ hơn

        return feature_maps,


# Khởi tạo mô hình ViT
# vit_model = ViT(img_dim=224, patch_dim=16, num_channels=3, num_classes=3, embed_dim=768, num_heads=12, num_encoder_layers=12)