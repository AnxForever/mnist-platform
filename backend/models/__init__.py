# 模型模块 - 包含所有深度学习模型定义 
import torch
import torch.nn as nn
from .mlp import MLP
from .cnn import CNN
from .rnn import RNN
from .attention_layers import SimpleAttention

def get_model_instance(model_id: str, has_attention: bool = False):
    """
    获取模型实例。这是一个工厂函数，根据模型ID返回相应的模型对象。
    """
    # 定义不需要 has_attention 参数的基础模型
    base_models = {
        'mlp': MLP,
        'cnn': CNN,
        'rnn': RNN
    }
    
    # 定义需要 has_attention 参数的注意力模型
    attention_models = {
        'mlp_attention': MLP,
        'cnn_attention': CNN,
        'rnn_attention': RNN
    }

    if model_id in base_models:
        # 如果是基础模型，直接实例化，不传递 has_attention
        model_class = base_models[model_id]
        print(f"🔧 正在实例化基础模型: {model_class.__name__}")
        return model_class()
    elif model_id in attention_models:
        # 如果是注意力模型，实例化时传递 has_attention=True
        model_class = attention_models[model_id]
        print(f"🔧 正在实例化注意力模型: {model_class.__name__} with Attention")
        return model_class(has_attention=True)
    else:
        print(f"❌ 未知的模型ID: {model_id}")
        raise ValueError(f"未知的模型ID: {model_id}") 