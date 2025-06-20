import os
import json
import torch
from datetime import datetime

class PersistenceManager:
    """
    处理所有文件系统操作，包括加载/保存模型、检查点和训练历史。
    确保所有文件操作都是线程安全的。
    """
    def __init__(self, base_dir, lock):
        self.base_dir = base_dir
        self.history_file = os.path.join(base_dir, 'training_history.json')
        self.checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        self.saved_models_dir = os.path.join(base_dir, 'saved_models')
        self.lock = lock

        # 确保目录存在
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.saved_models_dir, exist_ok=True)

    def save_training_history(self, new_entry):
        """
        以线程安全的方式，将一条新的训练记录追加到历史文件中。
        """
        with self.lock:
            history = self.load_training_history_internal()
            history.append(new_entry)
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=4)
            except IOError as e:
                print(f"❌ 无法写入训练历史文件: {e}")

    def load_training_history(self):
        """
        以线程安全的方式，加载完整的训练历史。
        """
        with self.lock:
            return self.load_training_history_internal()

    def load_training_history_internal(self):
        """
        内部加载方法，不包含锁，供其他带锁方法调用。
        """
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                # 处理文件为空的情况
                content = f.read()
                if not content:
                    return []
                return json.loads(content)
        except (IOError, json.JSONDecodeError) as e:
            print(f"⚠️ 无法加载或解析训练历史文件，将返回空列表: {e}")
            return []

    def clear_training_history(self):
        """
        清空训练历史文件。
        """
        with self.lock:
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                print("🧹 训练历史已清空。")
            except IOError as e:
                print(f"❌ 无法清空训练历史文件: {e}")


    def save_checkpoint(self, model, model_id, job_id, epoch, accuracy):
        """
        保存训练检查点。文件名现在只与 job_id 相关，以实现覆盖保存。
        这确保了每个训练任务只在磁盘上保留一个最新的"最佳"检查点。
        """
        filename = f"{job_id}_best.pth"
        filepath = os.path.join(self.checkpoints_dir, filename)
        with self.lock:
            try:
                torch.save(model.state_dict(), filepath)
                print(f"💾 已更新最佳检查点: {filepath} (Epoch: {epoch}, Accuracy: {accuracy:.4f})")
            except IOError as e:
                print(f"❌ 保存检查点失败: {e}")

    def save_final_model(self, model, model_id, best_accuracy):
        """
        训练结束后，保存最终的最佳模型。
        只有当当前模型的准确率高于已保存的同类型最佳模型时，才会保存。
        """
        with self.lock:
            # 1. 扫描现有同类型最佳模型，找出最高准确率
            highest_existing_acc = 0.0
            existing_best_models = []
            for f in os.listdir(self.saved_models_dir):
                if f.startswith(model_id + '_best_acc_') and f.endswith('.pth'):
                    existing_best_models.append(f)
                    try:
                        # 从文件名 'model-id_best_acc_0.9935.pth' 中提取 '0.9935'
                        acc_str = f.replace(model_id + '_best_acc_', '').replace('.pth', '')
                        acc = float(acc_str)
                        if acc > highest_existing_acc:
                            highest_existing_acc = acc
                    except (ValueError, IndexError):
                        # 如果文件名格式不正确，就忽略它
                        print(f"⚠️ 警告：无法从文件名 {f} 解析准确率。")
                        continue

            # 2. 比较准确率
            if best_accuracy > highest_existing_acc:
                print(f"🏆 新纪录！当前模型准确率 {best_accuracy:.4f} > 已有最高准确率 {highest_existing_acc:.4f}。")
                
                # 3. 删除所有旧的同类型最佳模型
                for f_to_delete in existing_best_models:
                    try:
                        os.remove(os.path.join(self.saved_models_dir, f_to_delete))
                        print(f"🗑️ 已删除旧的最佳模型: {f_to_delete}")
                    except OSError as e:
                        print(f"❌ 删除旧模型 {f_to_delete} 失败: {e}")

                # 4. 保存新的最佳模型
                filename = f"{model_id}_best_acc_{best_accuracy:.4f}.pth"
                filepath = os.path.join(self.saved_models_dir, filename)
                try:
                    torch.save(model.state_dict(), filepath)
                    print(f"🏅 已保存新的最佳模型: {filepath}")
                except IOError as e:
                    print(f"❌ 保存最终模型失败: {e}")
            else:
                print(f"📉 无需更新。当前模型准确率 {best_accuracy:.4f} 未超过已有的最佳模型 (准确率 {highest_existing_acc:.4f})。")

    def get_latest_checkpoint(self, model_id):
        """
        查找并返回指定模型ID的最新检查点文件路径。
        """
        with self.lock:
            checkpoints = [f for f in os.listdir(self.checkpoints_dir) if f.startswith(model_id) and f.endswith('.pth')]
            if not checkpoints:
                return None
            
            # 解析文件名以查找最新的epoch
            latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_epoch_')[1].split('_acc_')[0]))
            return os.path.join(self.checkpoints_dir, latest_checkpoint) 