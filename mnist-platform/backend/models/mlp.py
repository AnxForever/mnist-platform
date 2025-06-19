# MLP (多层感知机) 基础模型
import torch.nn as nn
from .base_model import BaseModel
from .attention_layers import SimpleAttention

class MLP(BaseModel):
    """多层感知机模型，可选注意力机制"""
    
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, has_attention=False):
        super(MLP, self).__init__()
        self.has_attention = has_attention
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        # 注意力层 - 在第二个隐藏层之后添加
        if self.has_attention:
            self.attention = SimpleAttention(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        
        # 应用注意力机制
        if self.has_attention:
            x = self.attention(x)
            
        x = self.fc3(x)
        return x 