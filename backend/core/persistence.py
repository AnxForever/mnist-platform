# 持久化模块 - 读写 JSON 历史和模型文件
import json
import os
import threading
import torch
import time
from datetime import datetime

# 全局锁，保护文件写操作
FILE_LOCK = threading.Lock()

class PersistenceManager:
    """
    负责所有与文件系统相关的持久化操作，例如保存和加载训练历史。
    这个类的目标是集中管理所有I/O操作，使其更加健壮和易于维护。
    """
    def __init__(self, base_dir, lock):
        """
        初始化持久化管理器。
        Args:
            base_dir (str): 所有持久化文件的根目录。
            lock (threading.Lock): 用于同步文件访问的线程锁。
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "saved_models")
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.history_file = os.path.join(base_dir, "training_history.json")
        self.lock = lock
        
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
    
    def save_training_history(self, new_entry):
        """
        以线程安全的方式，将一条新的训练记录追加到历史文件中。
        """
        with self.lock:
            try:
                # 1. 读取现有数据
                if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                else:
                    history = []
                
                # 2. 追加新记录
                history.append(new_entry)
                
                # 3. 写回文件
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                
                print(f"💾 训练历史已保存, Job ID: {new_entry.get('job_id')}")

            except (IOError, json.JSONDecodeError) as e:
                print(f"❌ 保存训练历史失败: {e}")
    
    def load_training_history(self):
        """
        以线程安全的方式，从文件加载完整的训练历史。
        """
        with self.lock:
            try:
                if not os.path.exists(self.history_file):
                    return []
                
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    # 添加一个检查，如果文件为空，则返回空列表，防止json.load抛出异常
                    content = f.read()
                    if not content:
                        return []
                    return json.loads(content)
            
            except (IOError, json.JSONDecodeError) as e:
                print(f"❌ 加载训练历史失败: {e}")
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