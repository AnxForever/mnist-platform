# 模型基类 - 所有模型必须继承此类
import torch.nn as nn

class BaseModel(nn.Module):
    """所有模型的基类，定义通用接口"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        """前向传播方法 - 必须被子类实现"""
        raise NotImplementedError("子类必须实现 forward 方法")
    
    def count_parameters(self):
        """统计模型总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """返回模型基本信息"""
        return {
            "parameter_count": self.count_parameters(),
            "model_type": self.__class__.__name__
        } 