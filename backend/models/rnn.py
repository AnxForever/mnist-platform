import torch
import torch.nn as nn
from .base_model import BaseModel
from .attention_layers import Attention

class RNN(BaseModel):
    def __init__(self, use_attention=False):
        super(RNN, self).__init__()
        self.use_attention = use_attention
        
        # 将图像的每一行视为一个时间步
        self.input_size = 28  # 每行28个像素
        self.hidden_size = 128
        self.num_layers = 2
        
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        if self.use_attention:
            self.attention = Attention(feature_dim=self.hidden_size, step_dim=28)
            
        self.fc = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        # x 的原始形状: (batch_size, 1, 28, 28)
        # 调整为 (batch_size, 28, 28) 以匹配RNN的输入序列
        x = x.squeeze(1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN 输出: (batch_size, seq_len, hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        
        if self.use_attention:
            # 使用注意力机制对所有时间步的输出进行加权
            out, _ = self.attention(out)
        else:
            # 只取最后一个时间步的输出
            out = out[:, -1, :]
            
        out = self.fc(out)
        return out

    def get_name(self):
        return "rnn_attention" if self.use_attention else "rnn" 