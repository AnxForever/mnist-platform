# Flask 应用主入口文件
# 定义所有 API 路由

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import os
import uuid
import time as time_module  # 使用别名避免潜在的命名冲突
import threading
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import numpy as np
import json
import re
from datetime import datetime
from werkzeug.utils import secure_filename

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# PIL for image processing
try:
    from PIL import Image
except ImportError:
    print("⚠️ PIL库未安装，手写识别功能可能无法正常工作")
    Image = None

# Project-specific imports
from models import get_model_instance
from core.persistence import PersistenceManager

app = Flask(__name__)
CORS(app)

# --- 路径配置 (Path Configuration) ---
# 使用 __file__ 获取 app.py 的绝对路径，确保路径的准确性
# os.path.dirname(...) 获取文件所在的目录 (即 backend 目录)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 基于 backend 目录，构建模型和检查点目录的绝对路径
SAVED_MODELS_DIR = os.path.join(APP_DIR, 'saved_models')
CHECKPOINTS_DIR = os.path.join(APP_DIR, 'checkpoints')

# 配置Flask应用以正确处理中文JSON输出
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# 全局状态管理
TRAINING_JOBS = {}
LOADED_MODELS = {}
# 允许通过环境变量配置最大并发数，默认为5
MAX_CONCURRENT_TRAINING_JOBS = int(os.environ.get('MAX_CONCURRENT_TRAINING_JOBS', 5))
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRAINING_JOBS)
TRAINING_LOCK = threading.Lock()
# 将持久化目录固定到 backend 目录，并将锁注入管理器
PERSISTENCE_MANAGER = PersistenceManager(APP_DIR, lock=TRAINING_LOCK)

# 6个模型的配置信息
AVAILABLE_MODELS = [
    {
        "id": "mlp",
        "name": "MLP (多层感知机)",
        "description": "最简单的全连接神经网络",
        "has_attention": False,
        "parameter_count": 79510
    },
    {
        "id": "cnn",
        "name": "CNN (卷积神经网络)",
        "description": "专为图像处理设计的经典架构",
        "has_attention": False,
        "parameter_count": 94214
    },
    {
        "id": "rnn",
        "name": "RNN (循环神经网络)",
        "description": "将图像按行序列化处理的实验性方法",
        "has_attention": False,
        "parameter_count": 127626
    },
    {
        "id": "mlp_attention",
        "name": "MLP + Attention",
        "description": "在MLP基础上加入注意力机制",
        "has_attention": True,
        "parameter_count": 85638
    },
    {
        "id": "cnn_attention",
        "name": "CNN + Attention",
        "description": "在CNN基础上加入注意力机制，实现空间注意力",
        "has_attention": True,
        "parameter_count": 102350
    },
    {
        "id": "rnn_attention",
        "name": "RNN + Attention",
        "description": "在RNN基础上加入注意力机制，增强序列建模能力",
        "has_attention": True,
        "parameter_count": 134792
    }
]

# 新增工具函数，方便通过ID查找名称
def get_model_name_by_id(model_id):
    """根据模型ID获取模型显示名称"""
    for model in AVAILABLE_MODELS:
        if model['id'] == model_id:
            return model['name']
    return model_id # 如果没找到，返回原始ID

@app.route('/api/status', methods=['GET'])
def get_status():
    """检查服务器状态"""
    return jsonify({
        "status": "running",
        "message": "MNIST智能分析平台后端服务正常运行",
        "timestamp": time_module.strftime("%Y-%m-%dT%H:%M:%S")
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可选模型列表"""
    try:
        # 使用json.dumps确保中文字符正确输出
        json_str = json.dumps(AVAILABLE_MODELS, ensure_ascii=False, indent=2)
        return Response(json_str, mimetype='application/json; charset=utf-8')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def start_training():
    """启动模型训练"""
    try:
        data = request.get_json()
        if not data or 'models' not in data:
            return jsonify({"error": "请求体格式错误，需要包含models字段"}), 400
        
        training_configs = data['models']
        jobs = []
        
        for config in training_configs:
            # 生成唯一的job_id
            job_id = f"job_{config['id']}_{int(time_module.time() * 1000)}"
            
            # 创建任务信息
            job_info = {
                "job_id": job_id,
                "model_id": config['id'],
                "status": "queued",
                "start_time": time_module.time(),
                "end_time": None,
                "progress": {
                    "percentage": 0,
                    "current_epoch": 0,
                    "total_epochs": config.get('epochs', 10),
                    "accuracy": 0.0,
                    "loss": 0.0,
                    "best_accuracy": 0.0,
                    "samples_per_second": 0,
                    "historical_best_accuracy": 0.0,
                    "is_first_training": True
                },
                "config": {
                    "epochs": config.get('epochs', 10),
                    "learning_rate": config.get('lr', 0.001),
                    "batch_size": config.get('batch_size', 64)
                },
                "error_message": None
            }
            
            # 保存到全局状态
            TRAINING_JOBS[job_id] = job_info
            
            jobs.append({
                "job_id": job_id,
                "model_id": config['id']
            })
            
            # 启动实际的训练任务
            TRAINING_EXECUTOR.submit(safe_training_wrapper, job_id, config['id'], config.get('epochs', 10), config.get('lr', 0.001), config.get('batch_size', 64))
        
        return jsonify({"jobs": jobs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def safe_training_wrapper(job_id, model_id, epochs, lr, batch_size):
    """安全的训练包装函数，包含完整的错误处理"""
    try:
        print(f"🚀 启动训练任务 {job_id}：模型 {model_id}, Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
        
        # 更新状态为运行中
        job_info = None
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                job_info = TRAINING_JOBS[job_id]
                job_info['status'] = 'running'
                job_info['start_time'] = time_module.time()
        
        # 执行真实训练并捕获其返回的完整结果
        results = perform_real_training(job_id, model_id, epochs, lr, batch_size)
        
        # 准备用于持久化的历史记录条目 - 完全按照前端ui.js的需求来构建
        end_time = time_module.time()
        start_time = job_info.get('start_time', end_time)

        history_entry = {
            "job_id": job_id,
            "model_id": model_id,
            "model_name": get_model_name_by_id(model_id),
            "status": "completed",
            "timestamp": datetime.fromtimestamp(end_time).isoformat(), # 前端需要 timestamp
            "config": job_info.get('config', {}),
            
            # --- 直接暴露在顶层的指标 ---
            "best_accuracy": results.get('best_accuracy', 0.0), # 原始准确率 (0.0 to 1.0)
            "final_loss": results.get('final_loss', 0.0),
            "samples_per_second": results.get('samples_per_second', 0),
            "model_params": results.get('model_params', 0),
            "duration_seconds": end_time - start_time, # 前端需要 duration_seconds

            "epoch_metrics": results.get('epoch_metrics', []),
            "error_message": None
        }

        PERSISTENCE_MANAGER.save_training_history(history_entry)
        
        print(f"✅ 训练任务 {job_id} 成功完成，结果已按照前端格式保存。")
        
    except Exception as e:
        error_message = f"训练失败: {str(e)}"
        print(f"❌ 训练任务 {job_id} 失败: {error_message}")
        
        # 更新错误状态
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'error'
                TRAINING_JOBS[job_id]['error_message'] = error_message
                TRAINING_JOBS[job_id]['end_time'] = time_module.time()

def run_training_job(job_id, model_id, epochs, lr, batch_size):
    """在后台线程中运行的训练任务 - 已弃用，使用 safe_training_wrapper"""
    pass

def is_first_training_for_model(model_id, save_dir):
    """检查是否为某模型的首次训练
    
    Args:
        model_id: 模型ID（如'cnn', 'mlp'等）
        save_dir: 模型保存目录路径
    
    Returns:
        tuple: (is_first_training: bool, historical_best_accuracy: float)
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            return True, 0.0
        
        # 扫描已有模型文件
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
        
        # 查找相同模型类型的文件
        matching_files = []
        for filename in model_files:
            if filename.startswith(f"{model_id}_"):
                matching_files.append(filename)
        
        if not matching_files:
            return True, 0.0
        
        # 提取历史最佳准确率
        best_accuracy = 0.0
        for filename in matching_files:
            accuracy = extract_accuracy_from_filename(filename)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        return False, best_accuracy
        
    except Exception as e:
        print(f"⚠️ 检查首次训练状态失败: {e}")
        return True, 0.0

def extract_accuracy_from_filename(filename):
    """从文件名中提取准确率
    
    Args:
        filename: 模型文件名
    
    Returns:
        float: 准确率，解析失败时返回0.0
    """
    try:
        # 支持两种格式：
        # 1. model_id_best_acc_0.9947.pth
        # 2. model_id_20250618_232011_acc_0.9947.pth
        
        if '_acc_' in filename:
            # 找到最后一个 '_acc_'
            last_acc_index = filename.rfind('_acc_')
            accuracy_part = filename[last_acc_index + 5:].replace('.pth', '')
            return float(accuracy_part)
        
        return 0.0
        
    except (ValueError, IndexError):
        return 0.0

def save_model_with_replacement(model, model_id, accuracy, save_dir, job_id):
    """保存模型，如果性能更好则替换旧模型
    
    Args:
        model: 要保存的PyTorch模型
        model_id: 模型ID
        accuracy: 当前模型的准确率
        save_dir: 保存目录
        job_id: 训练任务ID
    """
    try:
        # 检查是否为首次训练
        is_first, historical_best = is_first_training_for_model(model_id, save_dir)
        
        # 构建新的文件名（标准格式）
        new_filename = f"{model_id}_best_acc_{accuracy:.4f}.pth"
        new_filepath = os.path.join(save_dir, new_filename)
        
        # 如果不是首次训练，需要检查是否超越历史最佳
        if not is_first:
            if accuracy <= historical_best:
                print(f"⚠️ 当前准确率 {accuracy:.4f} 未超越历史最佳 {historical_best:.4f}")
                return False
            
            # 删除旧的模型文件
            model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth') and f.startswith(f"{model_id}_")]
            for old_file in model_files:
                old_filepath = os.path.join(save_dir, old_file)
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
                    print(f"🗑️ 删除旧模型文件: {old_file}")
        
        # 保存新模型
        torch.save(model.state_dict(), new_filepath)
        print(f"💾 模型保存成功: {new_filename}")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   文件大小: {os.path.getsize(new_filepath) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
        return False

def find_optimal_batch_size(model, device, base_batch_size=64):
    """动态寻找最优批量大小
    
    Args:
        model: PyTorch模型
        device: 设备 (cuda/cpu)
        base_batch_size: 基础批量大小
    
    Returns:
        int: 最优批量大小
    """
    if device.type == 'cpu':
        return min(base_batch_size, 32)  # CPU限制更严格
    
    optimal_batch_size = base_batch_size
    
    try:
        # 尝试逐步增加批量大小直到内存不足
        for test_batch_size in [64, 128, 256, 512]:
            try:
                # 创建测试数据
                test_input = torch.randn(test_batch_size, 1, 28, 28).to(device)
                
                # 前向传播测试
                model.eval()
                with torch.no_grad():
                    _ = model(test_input)
                
                optimal_batch_size = test_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e
    except Exception as e:
        print(f"⚠️ 批量大小优化失败，使用默认值: {e}")
    
    return optimal_batch_size

def perform_real_training(job_id, model_id, epochs, lr, batch_size):
    """执行实际的模型训练，并在完成后返回一个包含所有结果的字典。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔩 使用设备: {device}")

    # --- 路径设置 ---
    # 使用在文件顶部定义的全局路径常量
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # 获取历史最佳准确率
    is_first_train, historical_best_accuracy = is_first_training_for_model(model_id, SAVED_MODELS_DIR)
    
    with TRAINING_LOCK:
        TRAINING_JOBS[job_id]['progress']['historical_best_accuracy'] = historical_best_accuracy
        TRAINING_JOBS[job_id]['progress']['is_first_training'] = is_first_train

    print(f"📚 模型 {model_id} 的历史最佳准确率: {historical_best_accuracy:.4f}")

    train_start_time = time_module.time()
    model = None
    final_accuracy = 0.0
    is_new_record = False
    
    try:
        print(f"🎯 开始真实训练: {model_id}")
        
        # 1. 设备检测、数据加载等...
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        model_instance = get_model_instance(model_id)
        model_instance.to(device)
        optimal_batch_size = find_optimal_batch_size(model_instance, device, batch_size)
        
        train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=optimal_batch_size, shuffle=False)
        
        model = get_model_instance(model_id).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_accuracy = 0.0
        
        # 2. 训练循环
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                batch_start_time = time_module.time()

                images, labels = images.to(device), labels.to(device)
                
                # 前向传播、计算损失、反向传播和优化
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                batch_end_time = time_module.time()
                batch_duration = batch_end_time - batch_start_time
                samples_per_second = len(images) / batch_duration if batch_duration > 0 else 0
                
                # 更新全局状态
                with TRAINING_LOCK:
                    if job_id in TRAINING_JOBS:
                        progress_data = TRAINING_JOBS[job_id]['progress']
                        progress_data['percentage'] = min(100, int(((epoch * len(train_loader) + i + 1) / (epochs * len(train_loader))) * 100))
                        progress_data['current_epoch'] = epoch + 1
                        progress_data['loss'] = loss.item()
                        progress_data['samples_per_second'] = samples_per_second
            
            # ... (测试阶段代码保持不变) ...
            model.eval()
            test_loss = 0.0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_accuracy = correct / len(test_loader.dataset)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                is_new_record = PERSISTENCE_MANAGER.save_checkpoint(model, model_id, job_id, epoch, best_accuracy)

            with TRAINING_LOCK:
                 if job_id in TRAINING_JOBS:
                    TRAINING_JOBS[job_id]['progress'].update({
                        'percentage': ((epoch + 1) / epochs) * 100,
                        'current_epoch': epoch + 1,
                        'accuracy': test_accuracy,
                        'best_accuracy': best_accuracy,
                    })

        final_accuracy = best_accuracy
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'completed'
                TRAINING_JOBS[job_id]['end_time'] = time_module.time()
        
    except Exception as e:
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'error'
                TRAINING_JOBS[job_id]['error_message'] = str(e)
                TRAINING_JOBS[job_id]['end_time'] = time_module.time()
    
    finally:
        # 3. 无论成功失败，都记录到历史档案
        training_duration = time_module.time() - train_start_time
        history_entry = {
            "job_id": job_id,
            "model_id": model_id,
            "model_name": get_model_name_by_id(model_id),
            "has_attention": get_model_display_info(model_id).get('has_attention', False),
            "status": TRAINING_JOBS.get(job_id, {}).get('status', 'unknown'),
            "start_time": train_start_time,
            "completion_time": time_module.time(),
            "hyperparameters": {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size
            },
            "metrics": {
                "final_accuracy": final_accuracy,
                "training_duration_sec": training_duration,
                "is_new_record": is_new_record,
                "total_params": sum(p.numel() for p in model.parameters()) if model else 0
            },
            "error_message": TRAINING_JOBS.get(job_id, {}).get('error_message')
        }
        PERSISTENCE_MANAGER.append_training_history(history_entry)
        print(f"H️⃣ 训练历史已存档: {job_id}")

        # 在所有epoch结束后
        # 保存最佳模型
        if best_accuracy > historical_best_accuracy:
            print(f"🎉 新纪录! 准确率从 {historical_best_accuracy:.4f} 提升到 {best_accuracy:.4f}。正在保存模型...")
            save_model_with_replacement(model, model_id, best_accuracy, SAVED_MODELS_DIR, job_id)
        else:
            print(f"👍 训练完成，但未超越历史最佳准确率({historical_best_accuracy:.4f})")

        # 返回包含所有需要信息的字典
        return {
            "best_accuracy": round(best_accuracy, 4),
            "final_loss": round(test_loss, 5),
            "total_time": training_duration,
            "samples_per_second": samples_per_second,
            "model_params": sum(p.numel() for p in model.parameters()) if model else 0
        }

@app.route('/api/history', methods=['GET'])
def get_training_history():
    """获取所有已完成的训练历史记录"""
    try:
        history = PERSISTENCE_MANAGER.load_training_history()
        # 在返回之前，为每条记录添加 model_name
        for record in history:
            record['model_name'] = get_model_name_by_id(record['model_id'])
        return jsonify(history)
    except Exception as e:
        # 如果文件不存在或解析失败，返回空列表
        if isinstance(e, FileNotFoundError):
            return jsonify([])
        print(f"❌ 获取训练历史失败: {e}")
        return jsonify({"error": "无法获取训练历史"}), 500

@app.route('/api/training_progress', methods=['GET'])
def get_training_progress():
    """获取训练进度"""
    try:
        job_ids = request.args.get('job_ids')
        if not job_ids:
            return jsonify({"error": "缺少job_ids参数"}), 400
        
        job_list = [job_id.strip() for job_id in job_ids.split(',')]
        
        progress_list = []
        
        with TRAINING_LOCK:
            for job_id in job_list:
                if job_id in TRAINING_JOBS:
                    job_info = TRAINING_JOBS[job_id].copy()
                    progress_list.append(job_info)
                else:
                    return jsonify({"error": f"训练任务 {job_id} 不存在"}), 404
        
        # 返回前端期望的格式
        return jsonify({"progress": progress_list})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/trained_models', methods=['GET'])
def get_trained_models():
    """获取已训练模型列表"""
    try:
        trained_models = scan_trained_models()
        return jsonify(trained_models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def scan_trained_models():
    """扫描模型保存目录，返回可用于预测的模型列表"""
    
    if not os.path.exists(SAVED_MODELS_DIR):
        print(f"⚠️ 模型目录 {SAVED_MODELS_DIR} 不存在，无法加载模型。")
        return []
        
    models = []
    # 正则表达式改进：更精确地匹配模型ID、准确率和可选的job_id
    # 分组: (1:model_id) (2:accuracy) (3:job_id, a non-capturing group for the 'job' prefix)
    pattern = re.compile(r"^(.*?_attention|cnn|mlp|rnn)_(\d+\.\d{2})%_acc(?:_job_.*)?\.pth$")

    for filename in os.listdir(SAVED_MODELS_DIR):
        match = pattern.match(filename)
        if match:
            model_id = match.group(1)
            accuracy_str = match.group(2)
            
            # 使用更稳健的方式查找模型显示名称和参数
            display_info = get_model_display_info(model_id)
            
            models.append({
                "model_id": model_id,
                "display_name": display_info["name"],
                "accuracy": float(accuracy_str),
                "parameter_count": display_info["parameter_count"],
                "filename": filename
            })

    # 按准确率降序排序
    models.sort(key=lambda x: x['accuracy'], reverse=True)
    return models

def get_model_display_info(model_id):
    """根据模型ID获取其显示名称和参数数量"""
    model_config = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    if model_config:
        return {
            "name": model_config["name"],
            "parameter_count": model_config.get("parameter_count", "N/A")
        }
    # 如果找不到，提供一个优雅的降级方案
    return {
        "name": model_id.replace("_", " ").title(),
        "parameter_count": "N/A"
    }

@app.route('/api/predict', methods=['POST'])
def predict():
    """处理手写数字识别请求"""
    try:
        data = request.get_json()
        if not data or 'model_id' not in data or 'image_base64' not in data:
            return jsonify({"error": "请求体格式错误，需要包含model_id和image_base64字段"}), 400
        
        model_id = data['model_id']
        image_base64 = data['image_base64']
        
        # 验证模型是否已加载
        if model_id not in LOADED_MODELS:
            print(f"⏳ 首次预测请求，正在加载模型: {model_id}")
            model = load_model_for_prediction(model_id)
            if model is None:
                return jsonify({"error": f"模型文件 '{model_id}' 不存在或无法加载"}), 404
            LOADED_MODELS[model_id] = model
        
        model = LOADED_MODELS[model_id]
        
        # 预处理图像
        input_tensor = preprocess_canvas_image(image_base64)
        
        # 执行推断
        prediction, probabilities = perform_inference(model, input_tensor)
        
        # 返回结果
        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities.tolist()
        })
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        # 在生产环境中，可以考虑记录更详细的错误日志
        return jsonify({"error": "预测过程中发生内部错误"}), 500

def load_model_for_prediction(model_id):
    """为预测加载指定的、最新的模型"""
    try:
        # 检查是否已缓存
        if model_id in LOADED_MODELS:
            print(f"📋 使用缓存的模型: {model_id}")
            return LOADED_MODELS[model_id]
        
        # 使用全局路径常量构建模型文件路径
        model_path = os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth")
        
        if not os.path.exists(model_path):
            print(f"❌ 预测模型文件不存在: {model_path}")
            return None
            
        print(f"같은 모델 로딩: {model_path}")

        # --- 智能解析模型类型 (Smartly Parse Model Type) ---
        # 从完整文件名中提取基础模型ID，例如从 'cnn_attention_best_acc_0.9947' 提取 'cnn_attention'
        delimiter_pattern = r'_(?:best_)?acc_'
        parts = re.split(delimiter_pattern, model_id)
        model_type = parts[0]

        print(f"🔄 正在加载基础模型: '{model_type}' (从 '{model_id}' 解析)")
        
        # 创建模型实例
        model_instance = get_model_instance(model_type)
        
        # 加载权重
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_instance.load_state_dict(checkpoint)
            
            # 设置为评估模式
            model_instance.eval()
            
            # 缓存模型（限制缓存数量避免内存过多）
            if len(LOADED_MODELS) >= 5:  # 最多缓存5个模型
                # 删除最旧的模型
                oldest_key = next(iter(LOADED_MODELS))
                del LOADED_MODELS[oldest_key]
                print(f"🗑️ 删除缓存模型: {oldest_key}")
            
            LOADED_MODELS[model_id] = model_instance
            print(f"✅ 模型加载成功并缓存: {model_id}")
            
            return model_instance
        except Exception as e:
            print(f"❌ 加载模型权重失败: {e}")
            return None
        
    except Exception as e:
        print(f"❌ 加载模型失败 {model_id}: {e}")
        return None

def preprocess_canvas_image(image_base64):
    """预处理Canvas图像数据
    
    Args:
        image_base64: base64编码的图像数据
    
    Returns:
        torch.Tensor: 预处理后的tensor，形状为(1, 1, 28, 28)，失败时返回None
    """
    try:
        # 解码base64图像
        if image_base64.startswith('data:image/'):
            # 移除data URL前缀
            image_base64 = image_base64.split(',')[1]
        
        # base64解码
        image_data = base64.b64decode(image_base64)
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 确保图像是灰度的
        image = image.convert('L')
        
        # 调整大小到 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 转换为PyTorch Tensor
        # 标准化参数必须和训练时完全一致！
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor = transform(image)
        
        # 添加一个批次维度 (C, H, W) -> (B, C, H, W)
        return tensor.unsqueeze(0)

    except Exception as e:
        # 打印更详细的错误
        import traceback
        traceback.print_exc()
        raise ValueError(f"无法处理base64图像数据: {e}")

def perform_inference(model, input_tensor):
    """执行模型推理
    
    Args:
        model: PyTorch模型
        input_tensor: 输入tensor
    
    Returns:
        tuple: (prediction, probabilities) - 预测结果和概率分布
    """
    try:
        with torch.no_grad():
            # 前向传播
            outputs = model(input_tensor)
            
            # 应用softmax获取概率分布
            probabilities = torch.softmax(outputs, dim=1)
            
            # 获取预测结果
            prediction = torch.argmax(probabilities, dim=1)
            
            # 转换为numpy数组
            probabilities = probabilities.squeeze().numpy()
            prediction = prediction.item()
            
            return prediction, probabilities
            
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        raise e

def check_system_health():
    """检查系统健康状态"""
    try:
        health_info = {
            "status": "healthy",
            "timestamp": time_module.strftime("%Y-%m-%dT%H:%M:%S"),
            "active_training_jobs": len([job for job in TRAINING_JOBS.values() if job['status'] == 'running']),
            "loaded_models": len(LOADED_MODELS),
            "available_models": len(AVAILABLE_MODELS)
        }
        
        return health_info
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time_module.strftime("%Y-%m-%dT%H:%M:%S")
        }

if __name__ == '__main__':
    print("🚀 启动MNIST智能分析平台后端服务...")
    print("🔧 系统健康检查...")
    
    health = check_system_health()
    print(f"✅ 系统状态: {health['status']}")
    print(f"📊 可用模型: {health['available_models']} 个")
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True) 