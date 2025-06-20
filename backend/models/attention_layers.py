import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    通用的注意力机制模块。
    可以用于 RNN 的输出，也可以用于 CNN 的特征图。
    """
    def __init__(self, feature_dim, step_dim):
        super(Attention, self).__init__()
        
        self.feature_dim = feature_dim
        self.step_dim = step_dim

        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        """
        x 的形状: (batch_size, step_dim, feature_dim)
        """
        # 计算注意力权重
        attn_weights = self.attention_net(x)  # (batch_size, step_dim, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 应用权重
        context_vector = torch.sum(attn_weights * x, dim=1) # (batch_size, feature_dim)
        return context_vector, attn_weights

class SpatialAttention(nn.Module):
    """
    空间注意力机制，用于 CNN 特征图。
    它会学习在特征图的空间维度上关注哪些区域。
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        """
        x 的形状: (batch_size, channels, height, width)
        """
        # 计算注意力图
        attn_map = self.conv2(F.relu(self.conv1(x))) # (batch_size, 1, height, width)
        attn_map = torch.sigmoid(attn_map)

        # 将注意力图广播并应用到原始特征图上
        return x * attn_map.expand_as(x) 