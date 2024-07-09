import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.model.classifier.H0 import H0
from src.data_preparation.ThyroidCancerDataset import ThyroidCancerDataset
from src.data_preparation.ThyroidCancerDataLoader import ThyroidCancerDataLoader

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  # Lấy toàn bộ trừ lớp FC và avgpool
    
    def forward(self, x):
        x = self.features(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_dim=2048, dim=256, num_classes=3):
        super(VisionTransformer, self).__init__()
        self.dim = dim

        self.flatten = nn.Flatten(2)
        self.linear_proj = nn.Linear(input_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional_encodings = nn.Parameter(torch.zeros(50, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (B, C, H, W) -> (B, C, H*W) -> (H*W, B, C)
        x = self.linear_proj(x)  # Linear projection (H*W, B, dim)
        
        cls_tokens = self.cls_token.expand(-1, B, -1)  # (1, B, dim)
        x = torch.cat((cls_tokens, x), dim=0)  # (1 + H*W, B, dim)
        
        # Cập nhật giá trị của positional_encodings
        self.positional_encodings.data = self.positional_encodings.data[0:x.size(0)]  # Cắt positional_encodings nếu cần thiết
        x += self.positional_encodings

        x = self.transformer(x)  # (1 + H*W, B, dim)
        x = self.fc(x[0])  # Lấy token cls và đưa qua FC layer
        x = self.output(x)  # (B, num_classes)
        return x

class H9(H0):
    def __init__(self, num_classes=3):
        super(H9, self).__init__()
        self.num_classes = num_classes
        self.resnet50_feature_extractor = ResNet50FeatureExtractor()
        self.vit = VisionTransformer(input_dim=2048, dim=256, num_classes=num_classes)
    
    def forward(self, x):
        x = self.resnet50_feature_extractor(x)
        x = self.vit(x)
        return x

    def get_feature_maps(self, x):
        """
        Sẽ chỉ trả về bản đồ đặc trưng sau khi qua ResNet50
        """
        x = self.resnet50_feature_extractor(x)
        return x

    def load_data(self, data_dir, classes={0: ["B2"], 1: ["B5"], 2: ["B6"]}):
        print('Creating dataset...')
        train_dataset = ThyroidCancerDataset(img_dir=data_dir, transform=None, classes=classes, balance=True, mode='train')
        print('Train dataset size:', train_dataset.__len__())
        valid_dataset = ThyroidCancerDataset(img_dir=data_dir, transform=None, classes=classes, balance=False, mode='valid')
        print('Valid dataset size:', valid_dataset.__len__())
        test_dataset = ThyroidCancerDataset(img_dir=data_dir, transform=None, classes=classes, balance=False, mode='test')
        print('Test dataset size:', test_dataset.__len__())

        print('Creating dataloader...')
        thyroidCancerDataLoader = ThyroidCancerDataLoader()
        train_loader = thyroidCancerDataLoader.get_data_loader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        print('Train loader size:', len(train_loader))
        valid_loader = thyroidCancerDataLoader.get_data_loader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
        print('Valid loader size:', len(valid_loader))
        test_loader = thyroidCancerDataLoader.get_data_loader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        print('Test loader size:', len(test_loader))

        return train_loader, valid_loader, test_loader
