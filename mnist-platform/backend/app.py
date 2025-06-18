# Flask 应用主入口文件
# 定义所有 API 路由

from flask import Flask, request, jsonify, Response
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

app = Flask(__name__)
CORS(app)

# 配置Flask应用以正确处理中文JSON输出
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# 全局状态管理
TRAINING_JOBS = {}
LOADED_MODELS = {}
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=3)
TRAINING_LOCK = threading.Lock()

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
                    "best_accuracy": 0.0
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
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'running'
                TRAINING_JOBS[job_id]['start_time'] = time_module.time()
        
        # 执行真实训练
        perform_real_training(job_id, model_id, epochs, lr, batch_size)
        
        print(f"✅ 训练任务 {job_id} 成功完成")
        
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
    """执行真实的模型训练
    
    Args:
        job_id: 训练任务ID
        model_id: 模型ID
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批量大小
    """
    try:
        print(f"🎯 开始真实训练: {model_id}")
        
        # 1. 设备检测
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {device}")
        
        # 2. 数据加载
        print("📊 加载MNIST数据集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        # 动态调整批量大小
        model_instance = get_model_instance(model_id)
        model_instance.to(device)
        
        optimal_batch_size = find_optimal_batch_size(model_instance, device, batch_size)
        print(f"📦 优化后批量大小: {optimal_batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=optimal_batch_size, shuffle=False)
        
        # 3. 模型、损失函数和优化器
        model = get_model_instance(model_id).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"🧠 模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 4. 训练循环
        best_accuracy = 0.0
        
        # 确保保存目录存在
        save_dir = 'backend/saved_models'
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            epoch_start_time = time_module.time()
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            epoch_start = time_module.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
                
                # 更新进度
                if batch_idx % 100 == 0:
                    progress_percentage = (epoch * len(train_loader) + batch_idx) / (epochs * len(train_loader)) * 100
                    
                    with TRAINING_LOCK:
                        if job_id in TRAINING_JOBS:
                            TRAINING_JOBS[job_id]['progress'].update({
                                'percentage': progress_percentage,
                                'current_epoch': epoch + 1,
                                'loss': loss.item(),
                                'accuracy': correct_train / total_train
                            })
            
            # 测试阶段
            model.eval()
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
            
            # 计算准确率
            train_accuracy = correct_train / total_train
            test_accuracy = correct_test / total_test
            epoch_time = time_module.time() - epoch_start
            
            print(f"📈 Epoch {epoch+1}/{epochs}:")
            print(f"   训练损失: {train_loss/len(train_loader):.4f}, 训练准确率: {train_accuracy:.4f}")
            print(f"   测试损失: {test_loss/len(test_loader):.4f}, 测试准确率: {test_accuracy:.4f}")
            print(f"   耗时: {epoch_time:.2f}秒")
            
            # 更新最佳准确率并保存模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                
                # 保存最佳模型
                save_success = save_model_with_replacement(model, model_id, test_accuracy, save_dir, job_id)
                if save_success:
                    print(f"🎉 新的最佳模型! 准确率: {test_accuracy:.4f}")
            
            # 更新训练进度
            with TRAINING_LOCK:
                if job_id in TRAINING_JOBS:
                    TRAINING_JOBS[job_id]['progress'].update({
                        'percentage': ((epoch + 1) / epochs) * 100,
                        'current_epoch': epoch + 1,
                        'accuracy': test_accuracy,
                        'loss': test_loss / len(test_loader),
                        'best_accuracy': best_accuracy
                    })
        
        # 训练完成
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'completed'
                TRAINING_JOBS[job_id]['end_time'] = time_module.time()
                TRAINING_JOBS[job_id]['progress']['percentage'] = 100
        
        print(f"🎊 训练完成! 最佳准确率: {best_accuracy:.4f}")
        
    except Exception as e:
        error_message = f"训练过程出错: {str(e)}"
        print(f"❌ {error_message}")
        
        # 更新错误状态
        with TRAINING_LOCK:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id]['status'] = 'error'
                TRAINING_JOBS[job_id]['error_message'] = error_message
                TRAINING_JOBS[job_id]['end_time'] = time_module.time()
        
        raise e

@app.route('/api/training_progress', methods=['GET'])
def get_training_progress():
    """获取训练进度"""
    try:
        job_ids = request.args.get('job_ids')
        if not job_ids:
            return jsonify({"error": "缺少job_ids参数"}), 400
        
        job_list = [job_id.strip() for job_id in job_ids.split(',')]
        
        progress_data = {}
        
        with TRAINING_LOCK:
            for job_id in job_list:
                if job_id in TRAINING_JOBS:
                    job_info = TRAINING_JOBS[job_id].copy()
                    progress_data[job_id] = job_info
                else:
                    return jsonify({"error": f"训练任务 {job_id} 不存在"}), 404
        
        return jsonify(progress_data)
        
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
    """扫描saved_models目录中的已训练模型
    
    Returns:
        list: 已训练模型列表，每个模型包含id、name、model_type、accuracy等信息
    """
    # 优先使用backend/saved_models目录
    if os.path.exists('backend/saved_models'):
        save_dir = 'backend/saved_models'
    elif os.path.exists('saved_models'):
        save_dir = 'saved_models'
    else:
        save_dir = 'backend/saved_models'  # 回退到标准位置
    trained_models = []
    
    try:
        if not os.path.exists(save_dir):
            print(f"📁 模型保存目录不存在: {save_dir}")
            return trained_models
        
        # 获取目录中的所有 .pth 文件
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
        
        if not model_files:
            print(f"📂 模型保存目录为空: {save_dir}")
            return trained_models
        
        print(f"🔍 扫描到 {len(model_files)} 个模型文件")
        
        for filename in model_files:
            try:
                # 解析文件名格式: {model_id}_best_acc_{accuracy:.4f}.pth
                if '_best_acc_' not in filename:
                    continue
                    
                # 找到最后一个 '_best_acc_' 来正确分割文件名
                last_acc_index = filename.rfind('_best_acc_')
                if last_acc_index == -1:
                    continue
                
                model_id = filename[:last_acc_index]
                accuracy_part = filename[last_acc_index + len('_best_acc_'):].replace('.pth', '')
                accuracy = float(accuracy_part)
                
                # 获取模型显示名称和信息
                model_info = get_model_display_info(model_id)
                if not model_info:
                    continue
                
                # 获取文件信息
                file_path = os.path.join(save_dir, filename)
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                
                # 格式化训练时间（使用time_module避免命名冲突）
                mtime = file_stats.st_mtime
                local_time = time_module.localtime(mtime)
                training_time = time_module.strftime('%Y-%m-%dT%H:%M:%S', local_time)
                
                trained_model = {
                    "id": filename.replace('.pth', ''),  # 使用完整文件名作为ID
                    "name": f"{model_info['name']} (准确率: {accuracy:.2%})",
                    "model_type": model_id,
                    "has_attention": model_info['has_attention'],
                    "accuracy": accuracy,
                    "training_time": training_time,
                    "file_size": file_size,
                    "parameter_count": model_info['parameter_count']
                }
                
                trained_models.append(trained_model)
                print(f"✅ 发现模型: {model_id} - 准确率: {accuracy:.4f}")
                
            except Exception as e:
                print(f"⚠️ 解析模型文件失败 {filename}: {e}")
                continue
        
        # 按准确率降序排序
        trained_models.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"📋 成功扫描 {len(trained_models)} 个有效的已训练模型")
        return trained_models
        
    except Exception as e:
        print(f"❌ 扫描已训练模型失败: {e}")
        return []

def get_model_display_info(model_id):
    """获取模型的显示信息
    
    Args:
        model_id: 模型ID（如'cnn', 'mlp'等）
    
    Returns:
        dict: 模型显示信息，如果模型ID无效则返回None
    """
    # 在全局模型配置中查找
    for model_config in AVAILABLE_MODELS:
        if model_config['id'] == model_id:
            return {
                'name': model_config['name'],
                'has_attention': model_config['has_attention'],
                'parameter_count': model_config['parameter_count']
            }
    
    print(f"⚠️ 未知的模型类型: {model_id}")
    return None

@app.route('/api/predict', methods=['POST'])
def predict():
    """执行手写识别"""
    try:
        data = request.get_json()
        if not data or 'model_id' not in data or 'image_base64' not in data:
            return jsonify({"error": "请求体格式错误，需要包含model_id和image_base64字段"}), 400
        
        model_id = data['model_id']
        image_base64 = data['image_base64']
        
        print(f"🔍 开始预测，模型: {model_id}")
        
        # 加载模型
        model = load_model_for_prediction(model_id)
        if model is None:
            return jsonify({"error": f"无法加载模型: {model_id}"}), 404
        
        # 预处理图像
        input_tensor = preprocess_canvas_image(image_base64)
        if input_tensor is None:
            return jsonify({"error": "图像预处理失败"}), 400
        
        # 执行推理
        prediction, probabilities = perform_inference(model, input_tensor)
        
        result = {
            "prediction": int(prediction),
            "probabilities": probabilities.tolist(),
            "confidence": float(probabilities.max())
        }
        
        print(f"✅ 预测完成: {prediction} (置信度: {probabilities.max():.4f})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

def load_model_for_prediction(model_id):
    """加载指定的模型用于预测
    
    Args:
        model_id: 模型ID（完整文件名，不含.pth扩展名）
    
    Returns:
        torch.nn.Module: 加载的模型，失败时返回None
    """
    try:
        # 检查是否已缓存
        if model_id in LOADED_MODELS:
            print(f"📋 使用缓存的模型: {model_id}")
            return LOADED_MODELS[model_id]
        
        # 构建模型文件路径，优先使用backend/saved_models
        if os.path.exists('backend/saved_models'):
            model_path = os.path.join('backend/saved_models', f"{model_id}.pth")
        elif os.path.exists('saved_models'):
            model_path = os.path.join('saved_models', f"{model_id}.pth")
        else:
            model_path = os.path.join('backend/saved_models', f"{model_id}.pth")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
        
        # 解析模型类型
        model_type = model_id.split('_best_acc_')[0]
        
        print(f"🔄 加载模型: {model_type} 从 {model_path}")
        
        # 创建模型实例
        model = get_model_instance(model_type)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        # 设置为评估模式
        model.eval()
        
        # 缓存模型（限制缓存数量避免内存过多）
        if len(LOADED_MODELS) >= 5:  # 最多缓存5个模型
            # 删除最旧的模型
            oldest_key = next(iter(LOADED_MODELS))
            del LOADED_MODELS[oldest_key]
            print(f"🗑️ 删除缓存模型: {oldest_key}")
        
        LOADED_MODELS[model_id] = model
        print(f"✅ 模型加载成功并缓存: {model_id}")
        
        return model
        
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
        
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')
        
        # 调整大小到28x28
        image = image.resize((28, 28), Image.LANCZOS)
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 归一化到[0,1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # 应用MNIST标准归一化（与训练时保持一致）
        image_array = (image_array - 0.1307) / 0.3081
        
        # 转换为PyTorch tensor
        image_tensor = torch.from_numpy(image_array)
        
        # 添加batch和channel维度: (28, 28) -> (1, 1, 28, 28)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        
        print(f"📷 图像预处理完成，tensor形状: {image_tensor.shape}")
        return image_tensor
        
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return None

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

@app.route('/api/history', methods=['GET'])
def get_training_history():
    """获取训练历史记录"""
    try:
        # 返回最近的训练任务记录
        with TRAINING_LOCK:
            history = []
            for job_id, job_info in TRAINING_JOBS.items():
                history.append({
                    "job_id": job_id,
                    "model_id": job_info['model_id'],
                    "status": job_info['status'],
                    "start_time": job_info['start_time'],
                    "end_time": job_info.get('end_time'),
                    "final_accuracy": job_info['progress'].get('best_accuracy', 0.0),
                    "config": job_info['config']
                })
            
            # 按开始时间降序排序
            history.sort(key=lambda x: x['start_time'], reverse=True)
            
            return jsonify(history)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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