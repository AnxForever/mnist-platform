import torch.nn as nn

class BaseModel(nn.Module):
    """
    所有模型的基类，强制实现 `get_name` 方法。
    这确保了每个模型都能被唯一地识别。
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        """
        所有子类都必须实现自己的前向传播逻辑。
        """
        raise NotImplementedError("每个模型都必须实现自己的 forward 方法。")

    def get_name(self):
        """
        返回模型的唯一标识符。
        """
        raise NotImplementedError("每个模型都必须实现 get_name 方法。") 