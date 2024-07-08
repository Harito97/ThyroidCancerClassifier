import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.classifier.H0 import H0

class H_ANN(H0):
    def __init__(self):
        super(H_ANN, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(3 * 224 * 224, 512)
        # Dropout layer to reduce overfitting
        self.dropout1 = nn.Dropout(0.1)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(512, 3)
    
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 3 * 224 * 224)
        # Pass the input through the first fully connected layer, then apply ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout1(x)
        # Pass the result through the second fully connected layer
        x = self.fc2(x)
        return x

    def get_feature_maps(self, x):
        return None