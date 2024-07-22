import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# from torchsummary import summary


class H97_ANN(nn.Module):
    def __init__(self):
        # Use this for data version 3
        super(H97_ANN, self).__init__()
        self.fc1 = nn.Linear(150528, 9)  # (3 * 224 * 224, 9)
        self.fc2 = nn.Linear(9, 7)
        self.fc3 = nn.Linear(7, 3)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 150528)  # (-1, 3 * 224 * 224)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # Softmax applied along the class dimension
        return x


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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt


class H97_EfficientNet(nn.Module):
    def __init__(self, num_classes: int = 3, retrainEfficientNet: bool = False):
        super(H97_EfficientNet, self).__init__()
        # Load a pretrained EfficientNet model
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
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

        # Hook for CAM
        self.features = None

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        self.features = x
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

    def get_cam(self, input_tensor, target_class=None):
        """
        Get the Class Activation Map for a given input and target class.

        Parameters:
        input_tensor (torch.Tensor): The input image tensor.
        target_class (int): The target class for which to visualize the CAM.

        Returns:
        cam (np.ndarray): The Class Activation Map.
        """
        # Forward pass to get the logits
        output = self.forward(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Get the weight of the last layer
        weights = self.fc4.weight[target_class].detach().cpu().numpy()

        # Get the features from the last conv layer
        features = self.features.squeeze().detach().cpu().numpy()

        # Calculate the CAM
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i] #, :, :]

        # Normalize the CAM
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam

    def plot_cam(self, input_image, cam, alpha=0.5):
        """
        Plot the Class Activation Map on top of the input image.

        Parameters:
        input_image (np.ndarray): The input image.
        cam (np.ndarray): The Class Activation Map.
        alpha (float): The transparency level for the CAM overlay.
        """
        plt.imshow(input_image)
        plt.imshow(cam, cmap="jet", alpha=alpha)
        plt.colorbar()
        plt.show()

# EG to use 
# # Tạo đối tượng model
# model = H97_EfficientNet(num_classes=3, retrainEfficientNet=False)

# # Load một batch hình ảnh
# images, labels = next(iter(train_loader))
# images = images.to(device)
# labels = labels.to(device)

# # Tính toán CAM cho hình ảnh đầu tiên trong batch
# input_image = images[0].unsqueeze(0)
# input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
# target_class = labels[0].item()

# # Lấy CAM
# model.eval()
# with torch.no_grad():
#     cam = model.get_cam(input_image, target_class)

# # Hiển thị CAM trên hình ảnh đầu vào
# model.plot_cam(input_image_np, cam)


# import torch
# import torch.nn as nn
# from transformers import ViTForImageClassification, ViTFeatureExtractor

# class ViTTinyModel(nn.Module):
#     def __init__(self, num_classes):
#         super(ViTTinyModel, self).__init__()
#         self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.num_layers = 12
#         self.initial_lr = 1e-4
#         self.lr_high = 10 * self.initial_lr
#         self.lr_low = self.initial_lr

#     def set_parameter_requires_grad(self, start_layer, end_layer):
#         """Set requires_grad for layers based on their index."""
#         layer_names = list(self.model.vit.named_parameters())
#         for i, (name, param) in enumerate(layer_names):
#             if i < start_layer:
#                 param.requires_grad = False
#             elif start_layer <= i < end_layer:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#     def get_optimizers(self, layer_idx):
#         """
#         Create different optimizers for different layers.
#         """
#         optimizers = []
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 # Split the parameter name based on '.' and check its prefix to determine the layer
#                 if 'encoder' in name:
#                     # Extract the layer index from the name
#                     try:
#                         # Example: name might be 'vit.encoder.layer.0.attention.self.query.weight'
#                         layer_number = int(name.split('encoder.layer.')[1].split('.')[0])
#                     except (IndexError, ValueError):
#                         layer_number = None

#                     if layer_number is not None:
#                         if layer_number >= layer_idx:
#                             optimizers.append({'params': param, 'lr': self.lr_high})
#                         else:
#                             optimizers.append({'params': param, 'lr': self.lr_low})
#                 else:
#                     # Handle other parameters if needed
#                     optimizers.append({'params': param, 'lr': self.lr_low})
#         return optimizers


#     def forward(self, images):
#         return self.model(images).logits


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms

# class ViT(nn.Module):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  in_channels=3,
#                  num_classes=10,
#                  dim=256,
#                  depth=12,
#                  heads=8,
#                  mlp_dim=512,
#                  dropout=0.1):
#         super(ViT, self).__init__()

#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2

#         # Patch embedding
#         self.patch_embedding = nn.Sequential(
#             nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
#             nn.Flatten(2),
#             nn.Linear(dim, dim)
#         )

#         # Positional encoding
#         self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

#         # Transformer blocks
#         self.transformer = nn.Sequential(
#             *[nn.TransformerEncoderLayer(
#                 d_model=dim,
#                 nhead=heads,
#                 dim_feedforward=mlp_dim,
#                 dropout=dropout
#             ) for _ in range(depth)]
#         )

#         # Classification head
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.fc = nn.Linear(dim, num_classes)

#     def forward(self, x):
#         # Extract patches and apply embedding
#         patches = self.patch_embedding(x)
#         patches = patches.permute(0, 2, 1).contiguous()

#         # Add positional encoding
#         batch_size = patches.size(0)
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_tokens, patches), dim=1)
#         x += self.positional_encoding

#         # Apply transformer
#         x = self.transformer(x)

#         # Classification head
#         cls_output = x[:, 0]
#         output = self.fc(cls_output)

#         return output
