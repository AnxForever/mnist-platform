# 持久化模块 - 读写 JSON 历史和模型文件
import json
import os
import threading
import torch
import time

# 全局锁，保护文件写操作
FILE_LOCK = threading.Lock()

class PersistenceManager:
    """持久化管理器 - 负责模型和历史数据的存储"""
    
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "saved_models")
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.history_file = os.path.join(base_dir, "training_history.json")
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def save_model(self, model, model_id, timestamp=None):
        """保存最终训练完成的模型"""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        model_filename = f"{model_id}_{timestamp}.pth"
        model_path = os.path.join(self.models_dir, model_filename)
        
        torch.save(model.state_dict(), model_path)
        return model_path
    
    def save_checkpoint(self, model, model_id, job_id, epoch, accuracy):
        """保存训练过程中的最佳检查点"""
        checkpoint_filename = f"{model_id}_{job_id}_best.pth"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_filename)
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'model_id': model_id,
            'job_id': job_id,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    
    def load_model(self, model_class, model_path):
        """加载模型"""
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    def append_training_history(self, history_entry):
        """追加训练历史记录"""
        with FILE_LOCK:
            # 读取现有历史
            history = self.get_training_history()
            
            # 添加新记录
            history.append(history_entry)
            
            # 写回文件
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
    
    def get_training_history(self):
        """获取所有训练历史记录"""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def get_saved_models(self):
        """获取所有已保存的模型文件信息"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pth'):
                model_info = self._parse_model_filename(filename)
                if model_info:
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def _parse_model_filename(self, filename):
        """解析模型文件名，提取信息"""
        try:
            # 格式: {model_id}_{timestamp}.pth
            name_without_ext = filename[:-4]  # 移除 .pth
            parts = name_without_ext.split('_')
            
            if len(parts) >= 3:  # model_id, date, time
                model_id = '_'.join(parts[:-2])  # 支持带下划线的model_id
                date_time = '_'.join(parts[-2:])
                
                return {
                    "id": filename[:-4],  # 完整文件名作为ID
                    "model_id": model_id,
                    "timestamp": date_time,
                    "filename": filename,
                    "has_attention": "attention" in model_id.lower()
                }
        except Exception:
            pass
        
        return None 