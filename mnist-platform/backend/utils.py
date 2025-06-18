# 工具函数模块
# 包含图像处理、数据转换等实用函数

import base64
import io
import numpy as np
import torch
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def base64_to_tensor(image_base64, target_size=(28, 28), invert_colors=True):
    """将base64图像转换为PyTorch tensor
    
    Args:
        image_base64 (str): base64编码的图像数据
        target_size (tuple): 目标尺寸，默认(28, 28)
        invert_colors (bool): 是否反转颜色（Canvas黑字白底 -> MNIST白字黑底）
    
    Returns:
        torch.Tensor: 形状为(1, 1, H, W)的tensor，失败时返回None
    """
    try:
        # 移除data URL前缀
        if image_base64.startswith('data:image/'):
            image_base64 = image_base64.split(',')[1]
        
        # base64解码
        image_data = base64.b64decode(image_base64)
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')
        
        # 调整大小
        image = image.resize(target_size, Image.LANCZOS)
        
        # 转换为numpy数组
        image_array = np.array(image, dtype=np.float32)
        
        # 反转颜色（如果需要）
        if invert_colors:
            image_array = 255 - image_array
        
        # 归一化到[0,1]
        image_array = image_array / 255.0
        
        # 转换为PyTorch tensor并添加维度
        tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        
        logger.info(f"图像转换成功，tensor形状: {tensor.shape}")
        return tensor
        
    except Exception as e:
        logger.error(f"base64转tensor失败: {e}")
        return None

def normalize_image(image_array, mean=0.1307, std=0.3081):
    """使用MNIST数据集的统计信息归一化图像
    
    Args:
        image_array (np.ndarray): 图像数组
        mean (float): MNIST数据集的均值
        std (float): MNIST数据集的标准差
    
    Returns:
        np.ndarray: 归一化后的图像数组
    """
    try:
        normalized = (image_array - mean) / std
        return normalized
    except Exception as e:
        logger.error(f"图像归一化失败: {e}")
        return image_array

def apply_transforms(tensor, mean=0.1307, std=0.3081):
    """应用MNIST标准变换
    
    Args:
        tensor (torch.Tensor): 输入tensor
        mean (float): 均值
        std (float): 标准差
    
    Returns:
        torch.Tensor: 变换后的tensor
    """
    try:
        # 应用标准化
        tensor = (tensor - mean) / std
        return tensor
    except Exception as e:
        logger.error(f"应用变换失败: {e}")
        return tensor

def tensor_to_image(tensor, denormalize=True, mean=0.1307, std=0.3081):
    """将tensor转换回PIL图像（用于调试）
    
    Args:
        tensor (torch.Tensor): 输入tensor，形状为(1, 1, H, W)或(H, W)
        denormalize (bool): 是否反归一化
        mean (float): 均值
        std (float): 标准差
    
    Returns:
        PIL.Image: 转换后的图像
    """
    try:
        # 移除batch和channel维度
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0).squeeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        
        # 转换为numpy
        array = tensor.detach().numpy()
        
        # 反归一化
        if denormalize:
            array = array * std + mean
        
        # 限制到[0,1]范围
        array = np.clip(array, 0, 1)
        
        # 转换到[0,255]
        array = (array * 255).astype(np.uint8)
        
        # 转换为PIL图像
        image = Image.fromarray(array, mode='L')
        
        return image
        
    except Exception as e:
        logger.error(f"tensor转图像失败: {e}")
        return None

def validate_image_tensor(tensor, expected_shape=(1, 1, 28, 28)):
    """验证图像tensor的格式
    
    Args:
        tensor (torch.Tensor): 要验证的tensor
        expected_shape (tuple): 期望的形状
    
    Returns:
        bool: 是否有效
    """
    try:
        if not isinstance(tensor, torch.Tensor):
            logger.error("输入不是torch.Tensor类型")
            return False
        
        if tensor.shape != expected_shape:
            logger.error(f"tensor形状不匹配，期望: {expected_shape}, 实际: {tensor.shape}")
            return False
        
        if torch.isnan(tensor).any():
            logger.error("tensor包含NaN值")
            return False
        
        if torch.isinf(tensor).any():
            logger.error("tensor包含无穷大值")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"tensor验证失败: {e}")
        return False

def format_file_size(size_bytes):
    """格式化文件大小显示
    
    Args:
        size_bytes (int): 文件大小（字节）
    
    Returns:
        str: 格式化的文件大小字符串
    """
    try:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    except Exception as e:
        logger.error(f"文件大小格式化失败: {e}")
        return f"{size_bytes} B"

def create_error_response(message, code=400):
    """创建标准错误响应
    
    Args:
        message (str): 错误消息
        code (int): HTTP状态码
    
    Returns:
        dict: 错误响应字典
    """
    return {
        "error": message,
        "status_code": code,
        "success": False
    }

def create_success_response(data, message="操作成功"):
    """创建标准成功响应
    
    Args:
        data: 响应数据
        message (str): 成功消息
    
    Returns:
        dict: 成功响应字典
    """
    return {
        "data": data,
        "message": message,
        "success": True
    }

# 导出所有函数
__all__ = [
    'base64_to_tensor',
    'normalize_image', 
    'apply_transforms',
    'tensor_to_image',
    'validate_image_tensor',
    'format_file_size',
    'create_error_response',
    'create_success_response'
]

logger.info("🔧 Utils 工具模块已加载") 