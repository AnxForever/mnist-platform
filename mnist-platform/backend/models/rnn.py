# RNN (循环神经网络) 基础模型
import torch
import torch.nn as nn
from .base_model import BaseModel

class RNN(BaseModel):
    """循环神经网络模型 - 将图像按行序列化处理"""
    
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        # 重塑为序列: (batch_size, 28, 28) - 28个时间步，每步28个特征
        x = x.squeeze(1)  # 移除通道维度
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 使用最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out 