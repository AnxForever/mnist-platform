# 模型模块 - 包含所有深度学习模型定义 
from .mlp import MLP
from .cnn import CNN
from .rnn import RNN

def get_model_instance(model_id):
    """
    根据模型ID获取模型实例。
    """
    model_map = {
        'mlp': MLP,
        'cnn': CNN,
        'rnn': RNN,
        'mlp_attention': lambda: MLP(has_attention=True),
        'cnn_attention': lambda: CNN(has_attention=True),
        'rnn_attention': lambda: RNN(has_attention=True),
    }
    
    constructor = model_map.get(model_id)
    if constructor:
        return constructor()
    else:
        raise ValueError(f"未知的模型 ID: {model_id}") 