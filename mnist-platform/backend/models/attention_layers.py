# 通用 Attention 模块实现
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """通用注意力层"""
    
    def __init__(self, input_dim, attention_dim=128):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(input_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        
    def forward(self, x):
        # TODO: 实现注意力机制
        # x shape: (batch_size, seq_len, input_dim) 或类似
        pass

class SpatialAttention(nn.Module):
    """空间注意力层 - 适用于CNN"""
    
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # TODO: 实现空间注意力
        pass

class SequenceAttention(nn.Module):
    """序列注意力层 - 适用于RNN"""
    
    def __init__(self, hidden_dim):
        super(SequenceAttention, self).__init__()
        # TODO: 实现序列注意力
        pass 

class SimpleAttention(nn.Module):
    """
    一个简单的自注意力层。
    它接收一个特征序列，并为序列的每个部分计算一个权重，
    然后返回加权的特征和。
    这有助于模型关注输入特征中最相关的部分。
    """
    def __init__(self, feature_dim):
        super(SimpleAttention, self).__init__()
        self.feature_dim = feature_dim
        # 用于计算注意力权重的简单线性层
        self.attention_fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        # 在这个简化的场景中, 我们假设x是一个扁平化的特征向量。
        # 为了应用注意力, 我们需要将其视为一个"序列"，这里我们将其视为长度为1的序列。
        # 这种注意力机制更适合序列数据, 但在这里我们将其应用于整个特征向量，
        # 学习一个标量权重来缩放整个特征向量。
        
        # 计算注意力分数 (batch_size, 1)
        attention_scores = self.attention_fc(x)
        
        # 将分数转换为概率分布 (这里只有一个值, 所以softmax后是1)
        # 实际上, 使用sigmoid会更合理，来学习一个0到1之间的门控权重
        attention_weights = torch.sigmoid(attention_scores)
        
        # 将权重应用到输入特征上
        # (batch_size, feature_dim) * (batch_size, 1) -> (batch_size, feature_dim)
        context_vector = x * attention_weights
        
        return context_vector 