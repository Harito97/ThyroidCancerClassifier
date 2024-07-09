import torch
import torch.nn as nn
import torchvision.models as models

class H97_ResNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(H97_ResNet, self).__init__()
        
        # Load pretrained ResNet50
        resnet50 = models.resnet50(pretrained=True)
        
        # Remove the fully connected layers and average pooling
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        
        # Add custom fully connected layers
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        
        x = self.fc2(x)
        x = nn.ReLU()(x)
        
        x = self.fc3(x)
        
        return x
