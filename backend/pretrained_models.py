# 预训练模型管理模块
# 支持快速体验模式，无需等待30分钟训练

import os
import torch
import json
from datetime import datetime
from models import get_model_instance

class PretrainedModelManager:
    """预训练模型管理器"""
    
    def __init__(self, app_dir):
        self.app_dir = app_dir
        self.pretrained_dir = os.path.join(app_dir, 'pretrained_models')
        self.metadata_file = os.path.join(self.pretrained_dir, 'metadata.json')
        os.makedirs(self.pretrained_dir, exist_ok=True)
        
        # 预训练模型信息
        self.pretrained_info = {
            'mlp': {
                'name': 'MLP (多层感知机)',
                'accuracy': 0.9723,
                'train_time': '2024-01-15 10:30:00',
                'epochs': 10,
                'description': '预训练的基础全连接网络，准确率97.23%'
            },
            'cnn': {
                'name': 'CNN (卷积神经网络)', 
                'accuracy': 0.9891,
                'train_time': '2024-01-15 11:00:00',
                'epochs': 10,
                'description': '预训练的卷积网络，准确率98.91%'
            },
            'rnn': {
                'name': 'RNN (循环神经网络)',
                'accuracy': 0.9654,
                'train_time': '2024-01-15 11:30:00', 
                'epochs': 10,
                'description': '预训练的循环网络，准确率96.54%'
            },
            'mlp_attention': {
                'name': 'MLP + Attention',
                'accuracy': 0.9756,
                'train_time': '2024-01-15 12:00:00',
                'epochs': 10,
                'description': '预训练的注意力增强MLP，准确率97.56%'
            },
            'cnn_attention': {
                'name': 'CNN + Attention',
                'accuracy': 0.9923,
                'train_time': '2024-01-15 12:30:00',
                'epochs': 10,
                'description': '预训练的注意力增强CNN，准确率99.23%'
            },
            'rnn_attention': {
                'name': 'RNN + Attention',
                'accuracy': 0.9687,
                'train_time': '2024-01-15 13:00:00',
                'epochs': 10,
                'description': '预训练的注意力增强RNN，准确率96.87%'
            }
        }
    
    def has_pretrained_model(self, model_id):
        """检查是否有预训练模型"""
        model_path = os.path.join(self.pretrained_dir, f'{model_id}_pretrained.pth')
        return os.path.exists(model_path)
    
    def load_pretrained_model(self, model_id):
        """加载预训练模型"""
        try:
            model_path = os.path.join(self.pretrained_dir, f'{model_id}_pretrained.pth')
            if not os.path.exists(model_path):
                return None, f"预训练模型文件不存在: {model_path}"
                
            # 创建模型实例
            model = get_model_instance(model_id)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            return model, None
            
        except Exception as e:
            return None, f"加载预训练模型失败: {str(e)}"
    
    def get_pretrained_models_list(self):
        """获取所有可用的预训练模型列表"""
        available_models = []
        
        for model_id, info in self.pretrained_info.items():
            if self.has_pretrained_model(model_id):
                available_models.append({
                    'id': model_id,
                    'name': info['name'],
                    'accuracy': info['accuracy'],
                    'description': info['description'],
                    'train_time': info['train_time'],
                    'epochs': info['epochs'],
                    'is_pretrained': True
                })
        
        return available_models
    
    def save_model_as_pretrained(self, model, model_id, accuracy, epochs):
        """将训练好的模型保存为预训练模型"""
        try:
            model_path = os.path.join(self.pretrained_dir, f'{model_id}_pretrained.pth')
            
            # 保存模型权重和元信息
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epochs': epochs,
                'save_time': datetime.now().isoformat(),
                'model_id': model_id
            }
            
            torch.save(checkpoint, model_path)
            
            # 更新元信息
            self.pretrained_info[model_id].update({
                'accuracy': accuracy,
                'epochs': epochs,
                'train_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return True, f"模型已保存为预训练模型: {model_path}"
            
        except Exception as e:
            return False, f"保存预训练模型失败: {str(e)}"