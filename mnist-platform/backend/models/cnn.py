# CNN (卷积神经网络) 基础模型
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .attention_layers import SimpleAttention

class CNN(BaseModel):
    """卷积神经网络模型，可选注意力机制"""
    
    def __init__(self, num_classes=10, has_attention=False):
        super(CNN, self).__init__()
        self.has_attention = has_attention
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 卷积层后的特征维度
        self.feature_dim = 128 * 3 * 3
        
        # 注意力层
        if self.has_attention:
            self.attention = SimpleAttention(self.feature_dim)
        
        # 全连接层
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.feature_dim)
        
        # 应用注意力机制
        if self.has_attention:
            x = self.attention(x)
            
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x 