import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .attention_layers import Attention

class MLP(BaseModel):
    def __init__(self, use_attention=False):
        super(MLP, self).__init__()
        self.use_attention = use_attention
        
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        if self.use_attention:
            # 注意：这里的step_dim=1, feature_dim=128
            self.attention = Attention(feature_dim=128, step_dim=1)
        
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        
        if self.use_attention:
            # 为了使用通用的Attention,需要增加一个step维度
            x, _ = self.attention(x.unsqueeze(1))
            
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_name(self):
        return "mlp_attention" if self.use_attention else "mlp" 