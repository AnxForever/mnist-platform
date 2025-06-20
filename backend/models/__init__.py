def get_model_instance(model_id, **kwargs):
    """
    模型工厂函数，根据 model_id 动态导入并实例化模型。
    这样可以避免在主应用中硬编码所有模型导入。
    """
    if model_id == 'mlp':
        from .mlp import MLP
        return MLP(use_attention=False, **kwargs)
    elif model_id == 'cnn':
        from .cnn import CNN
        return CNN(use_attention=False, **kwargs)
    elif model_id == 'rnn':
        from .rnn import RNN
        return RNN(use_attention=False, **kwargs)
    elif model_id == 'mlp_attention':
        from .mlp import MLP
        return MLP(use_attention=True, **kwargs)
    elif model_id == 'cnn_attention':
        from .cnn import CNN
        return CNN(use_attention=True, **kwargs)
    elif model_id == 'rnn_attention':
        from .rnn import RNN
        return RNN(use_attention=True, **kwargs)
    else:
        raise ValueError(f"未知的模型ID: {model_id}")

# 可选：定义一个 __all__ 来明确包的公共API
__all__ = ['get_model_instance'] 