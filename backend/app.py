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
from torch.utils.data import DataLoader, random_split
import numpy as np

# PIL for image processing
try:
    from PIL import Image
except ImportError:
    print("⚠️ PIL库未安装，手写识别功能可能无法正常工作")
    Image = None

# Project-specific imports
from models import get_model_instance
from core.persistence import PersistenceManager
try:
    import psutil
except ImportError:
    psutil = None
    print("⚠️ 'psutil' a an pynvml 库未安装。")


app = Flask(__name__)
CORS(app)

# --- 路径配置 (Path Configuration) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(APP_DIR, 'saved_models')
CHECKPOINTS_DIR = os.path.join(APP_DIR, 'checkpoints')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


# 配置Flask应用以正确处理中文JSON输出
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# 全局状态管理
TRAINING_JOBS = {}
LOADED_MODELS = {}
MAX_CONCURRENT_TRAINING_JOBS = int(os.environ.get('MAX_CONCURRENT_TRAINING_JOBS', 5))
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRAINING_JOBS)
TRAINING_LOCK = threading.Lock()
PERSISTENCE_MANAGER = PersistenceManager(APP_DIR, lock=TRAINING_LOCK)

# 6个模型的配置信息
AVAILABLE_MODELS = [
    {"id": "mlp", "name": "MLP (多层感知机)", "description": "最简单的全连接神经网络", "has_attention": False, "parameter_count": 79510},
    {"id": "cnn", "name": "CNN (卷积神经网络)", "description": "专为图像处理设计的经典架构", "has_attention": False, "parameter_count": 94214},
    {"id": "rnn", "name": "RNN (循环神经网络)", "description": "将图像按行序列化处理的实验性方法", "has_attention": False, "parameter_count": 127626},
    {"id": "mlp_attention", "name": "MLP + Attention", "description": "在MLP基础上加入注意力机制", "has_attention": True, "parameter_count": 85638},
    {"id": "cnn_attention", "name": "CNN + Attention", "description": "在CNN基础上加入注意力机制，实现空间注意力", "has_attention": True, "parameter_count": 102350},
    {"id": "rnn_attention", "name": "RNN + Attention", "description": "在RNN基础上加入注意力机制，增强序列建模能力", "has_attention": True, "parameter_count": 134792}
]

def get_model_name_by_id(model_id):
    """根据模型ID获取模型显示名称"""
    for model in AVAILABLE_MODELS:
        if model['id'] == model_id:
            return model['name']
    return model_id

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
        return jsonify(AVAILABLE_MODELS)
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
            job_id = f"job_{config['id']}_{int(time_module.time() * 1000)}"
            epochs = config['epochs']
            lr = config['lr']
            batch_size = config['batch_size']
            
            job_info = {
                "job_id": job_id, "model_id": config['id'], "status": "queued",
                "start_time": time_module.time(), "end_time": None,
                "progress": {
                    "percentage": 0, "current_epoch": 0, "total_epochs": epochs,
                    "accuracy": 0.0, "loss": 0.0, "best_accuracy": 0.0, "samples_per_second": 0,
                },
                "config": {
                    "epochs": epochs, "learning_rate": lr,
                    "batch_size": batch_size
                },
                "error_message": None
            }
            TRAINING_JOBS[job_id] = job_info
            jobs.append({"job_id": job_id, "model_id": config['id']})
            TRAINING_EXECUTOR.submit(safe_training_wrapper, job_id, config['id'], 
                                     epochs, lr, batch_size)
        
        return jsonify({"jobs": jobs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def safe_training_wrapper(job_id, model_id, epochs, lr, batch_size):
    """安全的训练包装函数，包含完整的错误处理"""
    try:
        print(f"🚀 启动训练任务 {job_id}：模型 {model_id}, Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
        
        with TRAINING_LOCK:
            job_info = TRAINING_JOBS.get(job_id)
            if job_info:
                job_info['status'] = 'running'
                job_info['start_time'] = time_module.time()
        
        results = perform_real_training(job_id, model_id, epochs, lr, batch_size)
        
        end_time = time_module.time()
        start_time = job_info.get('start_time', end_time) if job_info else end_time

        history_entry = {
            "job_id": job_id, "model_id": model_id, "model_name": get_model_name_by_id(model_id),
            "status": "completed", "timestamp": datetime.fromtimestamp(end_time).isoformat(),
            "config": job_info.get('config', {}) if job_info else {},
            "best_accuracy": results.get('best_accuracy', 0.0),
            "final_train_loss": results.get('final_train_loss', 0.0),
            "final_val_loss": results.get('final_val_loss', 0.0),
            "samples_per_second": results.get('samples_per_second', 0),
            "duration_seconds": results.get('total_training_duration', end_time - start_time),
            "epoch_metrics": results.get('epoch_metrics', []),
            "stability_metrics": results.get('stability_metrics', {}),
            "environment_info": results.get('environment_info', {}),
            "hyperparameters_extended": results.get('hyperparameters_extended', {}),
            "error_message": None
        }
        PERSISTENCE_MANAGER.save_training_history(history_entry)
        
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'completed'
                TRAINING_JOBS[job_id]['end_time'] = end_time
        
        print(f"✅ 训练任务 {job_id} 成功完成。")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = f"训练失败: {str(e)}"
        print(f"❌ 训练任务 {job_id} 失败: {error_message}")
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'error'
                TRAINING_JOBS[job_id]['error_message'] = error_message
                TRAINING_JOBS[job_id]['end_time'] = time_module.time()

def collect_environment_info(device):
    return {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU',
        "cpu_count": os.cpu_count()
    }

def collect_model_architecture_info(model, model_id):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model_id": model_id,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layer_info": str(model)
    }

def monitor_system_resources():
    if not psutil:
        return {}
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_utilization_percent": get_gpu_utilization()
    }

def get_gpu_utilization():
    if not torch.cuda.is_available():
        return 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return util.gpu
    except (ImportError, Exception):
        return 0

def perform_real_training(job_id, model_id, epochs, lr, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{job_id}] 使用设备: {device}")

    environment_info = collect_environment_info(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(APP_DIR, 'data')
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    model = get_model_instance(model_id).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    architecture_info = collect_model_architecture_info(model, model_id)
    hyperparameters_extended = {
        "basic": {"epochs": epochs, "learning_rate": lr, "batch_size": batch_size},
        "optimizer": {"type": "Adam", "betas": optimizer.param_groups[0]['betas']},
        "model_architecture": architecture_info,
        "training": {"loss_function": "CrossEntropyLoss", "train_val_split": 0.9}
    }

    epoch_metrics = []
    best_val_accuracy = 0.0
    total_samples_processed = 0
    training_start_time = time_module.time()

    for epoch in range(epochs):
        epoch_start_time = time_module.time()
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        epoch_duration = time_module.time() - epoch_start_time
        samples_this_epoch = len(train_subset)
        total_samples_processed += samples_this_epoch
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            PERSISTENCE_MANAGER.save_checkpoint(
                model=model, model_id=model_id, job_id=job_id, 
                epoch=epoch + 1, accuracy=val_accuracy
            )

        current_metrics = {
            "epoch": epoch + 1, "val_loss": avg_val_loss, "val_accuracy": val_accuracy,
            "train_loss": avg_train_loss, "train_accuracy": train_accuracy,
            "epoch_duration_seconds": epoch_duration, 
            "samples_per_second": samples_this_epoch / epoch_duration if epoch_duration > 0 else 0
        }
        epoch_metrics.append(current_metrics)

        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['progress'].update({
                    "current_epoch": epoch + 1, "percentage": int(((epoch + 1) / epochs) * 100),
                    "accuracy": val_accuracy, "loss": avg_val_loss, "best_accuracy": best_val_accuracy,
                    "samples_per_second": current_metrics["samples_per_second"]
                })
        print(f"[{job_id}] Epoch {epoch+1}/{epochs} - Val Acc: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")

    total_training_duration = time_module.time() - training_start_time
    samples_per_second_overall = total_samples_processed / total_training_duration if total_training_duration > 0 else 0

    PERSISTENCE_MANAGER.save_final_model(model, model_id, best_val_accuracy)

    stability_metrics = {
        "accuracy_variance": np.var([m['val_accuracy'] for m in epoch_metrics]) if len(epoch_metrics) > 1 else 0.0,
        "loss_variance": np.var([m['val_loss'] for m in epoch_metrics]) if len(epoch_metrics) > 1 else 0.0
    }

    return {
        "best_accuracy": best_val_accuracy,
        "final_val_loss": epoch_metrics[-1]['val_loss'] if epoch_metrics else 0,
        "final_train_loss": epoch_metrics[-1]['train_loss'] if epoch_metrics else 0,
        "total_training_duration": total_training_duration,
        "samples_per_second": samples_per_second_overall,
        "epoch_metrics": epoch_metrics,
        "stability_metrics": stability_metrics,
        "environment_info": environment_info,
        "hyperparameters_extended": hyperparameters_extended
    }

@app.route('/api/history', methods=['GET'])
def get_training_history():
    """获取所有训练历史记录"""
    try:
        history = PERSISTENCE_MANAGER.load_training_history()
        return jsonify(history)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/training_progress', methods=['GET'])
def get_training_progress():
    """获取训练进度"""
    job_ids = request.args.get('job_ids')
    if not job_ids:
        return jsonify({"progress": []})  # Return empty list if no job_ids
    
    job_list = [job_id.strip() for job_id in job_ids.split(',')]
    progress_list = []
    with TRAINING_LOCK:
        for job_id in job_list:
            if job_id in TRAINING_JOBS:
                progress_list.append(TRAINING_JOBS[job_id].copy())
    return jsonify({"progress": progress_list})

@app.route('/api/trained_models', methods=['GET'])
def get_trained_models():
    """获取所有已保存的最佳模型及其详细信息。"""
    try:
        models = scan_trained_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def scan_trained_models():
    """
    扫描 saved_models 目录，返回每个最佳模型的详细信息列表。
    """
    trained_models = []
    # 正则表达式用于从文件名中提取 model_id 和 accuracy
    # 例如: cnn_attention_best_acc_0.9935.pth
    pattern = re.compile(r"^(?P<model_id>.+)_best_acc_(?P<accuracy>\d+\.\d+)\.pth$")
    
    if not os.path.exists(SAVED_MODELS_DIR):
        return []

    for filename in os.listdir(SAVED_MODELS_DIR):
        match = pattern.match(filename)
        if match:
            model_info = match.groupdict()
            model_id = model_info['model_id']
            accuracy = float(model_info['accuracy'])
            
            # 获取模型的显示名称
            display_name = get_model_name_by_id(model_id)

            trained_models.append({
                "model_id": model_id,
                "display_name": display_name,
                "filename": filename,
                "accuracy": accuracy
            })
            
    # 按准确率降序排序
    trained_models.sort(key=lambda x: x['accuracy'], reverse=True)
    return trained_models

def get_model_display_info(model_id):
    """根据模型ID获取其显示名称和参数数量"""
    model_config = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    return {"name": model_config["name"], "parameter_count": model_config.get("parameter_count", "N/A")} if model_config else {"name": model_id, "parameter_count": "N/A"}

@app.route('/api/predict', methods=['POST'])
def predict():
    """处理手写数字识别请求"""
    try:
        data = request.get_json()
        if not data or 'filename' not in data or 'image_base64' not in data:
            return jsonify({"error": "Request body must include filename and image_base64"}), 400
        
        filename = data['filename']
        model_id = filename.split('_best_acc_')[0]  # Infer model_id from filename

        if filename not in LOADED_MODELS:
            model = load_model_for_prediction(model_id, filename)
            if model is None:
                return jsonify({"error": f"Model file '{filename}' could not be loaded"}), 404
            LOADED_MODELS[filename] = model
        
        model = LOADED_MODELS[filename]
        input_tensor = preprocess_canvas_image(data['image_base64'])
        if input_tensor is None:
            return jsonify({"error": "Image preprocessing failed"}), 400
        prediction, probabilities = perform_inference(model, input_tensor)
        return jsonify({"predicted_class": prediction, "probabilities": probabilities.tolist()})
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return jsonify({"error": "Internal server error during prediction"}), 500

def load_model_for_prediction(model_id, filename):
    """为预测加载指定的模型文件。"""
    try:
        model_path = os.path.join(SAVED_MODELS_DIR, filename)
        if not os.path.exists(model_path):
            return None
        model_instance = get_model_instance(model_id)
        state_dict = torch.load(model_path, map_location='cpu')
        model_instance.load_state_dict(state_dict)
        model_instance.eval()
        return model_instance
    except Exception as e:
        print(f"❌ 加载模型 '{filename}' 失败: {e}")
        return None

def preprocess_canvas_image(image_base64):
    """预处理Canvas图像数据"""
    try:
        image_data = base64.b64decode(image_base64.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        if image.size != (28, 28):
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return None

def perform_inference(model, input_tensor):
    """执行模型推断"""
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        return prediction, probabilities

def check_system_health():
    """检查系统依赖和配置"""
    if not torch.cuda.is_available():
        print("\n" + "="*50 + "\nℹ️  提示: 未检测到 CUDA。模型将在 CPU 上训练，速度可能较慢。\n" + "="*50 + "\n")
    if not psutil:
        print("\n" + "="*50 + "\n⚠️  警告: 'psutil' 库未安装。系统资源监控将不可用。请运行: pip install psutil\n" + "="*50 + "\n")
    try:
        import pynvml
    except ImportError:
        print("\n" + "="*50 + "\n⚠️  警告: 'pynvml' 库未安装。GPU利用率监控将不可用。请运行: pip install nvidia-ml-py3\n" + "="*50 + "\n")


if __name__ == '__main__':
    check_system_health()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)