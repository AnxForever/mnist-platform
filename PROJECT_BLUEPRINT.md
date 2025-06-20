# MNIST 智能分析平台 - 项目核心蓝图

本文档是项目的核心参考，旨在明确前后端的接口约定、核心数据结构和关键模块职责，作为所有开发工作的"单一事实来源"(Single Source of Truth)。

---

## 1. 后端 API 接口清单 (`backend/app.py`)

### 1.1 `GET /api/models`
- **功能**: 获取所有可选的、用于训练的基础模型列表。
- **请求**: 无
- **响应 (JSON)**:
  ```json
  [
    {
      "id": "mlp",
      "name": "多层感知机 (MLP)"
    },
    {
      "id": "cnn",
      "name": "卷积神经网络 (CNN)"
    },
    // ... 其他模型
  ]
  ```

### 1.2 `POST /api/train`
- **功能**: 提交一个或多个模型的训练任务。
- **请求 (JSON)**:
  ```json
  {
    "model_ids": ["cnn_attention", "rnn_attention"], // 要训练的模型ID数组
    "config": {
      "epochs": 10,
      "learning_rate": 0.001,
      "batch_size": 64
    }
  }
  ```
- **响应 (JSON)**:
  ```json
  {
    "message": "训练任务已启动",
    "jobs": [
      {
        "job_id": "uuid-for-cnn-attention",
        "model_id": "cnn_attention"
      },
      {
        "job_id": "uuid-for-rnn-attention",
        "model_id": "rnn_attention"
      }
    ]
  }
  ```

### 1.3 `GET /api/training_history`
- **功能**: 获取所有已完成的训练任务历史记录。
- **响应 (JSON)**:
  ```json
  [ 
    // 包含多个 TrainingHistoryEntry 对象 (见 2.1)
  ]
  ```

### 1.4 `GET /api/training_history/<job_id>`
- **功能**: 获取指定 `job_id` 的单个训练任务的详细信息。
- **响应 (JSON)**:
  ```json
  // 单个 TrainingHistoryEntry 对象 (见 2.1)
  ```

### 1.5 `GET /api/training_progress`
- **功能**: 获取所有正在进行的训练任务的实时进度。
- **响应 (JSON)**:
  ```json
  {
    "active_jobs": [
      {
        "job_id": "uuid-for-cnn-attention",
        "model_id": "cnn_attention",
        "status": "training",
        "progress": {
            "current_epoch": 3,
            "percentage": 30,
            "accuracy": 0.9810,
            "loss": 0.08,
            "best_accuracy": 0.9815,
            "samples_per_second": 1200
        }
      }
    ],
    "completed_jobs_since_last_poll": []
  }
  ```
  
### 1.6 `GET /api/trained_models`
- **功能**: 获取所有已成功训练并保存了最优模型、可用于预测的模型列表。
- **响应 (JSON)**:
  ```json
  [
    {
      "id": "cnn_attention_best_acc_0.9935",
      "name": "CNN + Attention (准确率: 99.35%)"
    }
    // ... 其他训练好的模型
  ]
  ```

### 1.7 `POST /api/predict`
- **功能**: 对手写数字图片进行预测。
- **请求 (JSON)**:
  ```json
  {
    "model_id": "cnn_attention_best_acc_0.9935", // 从 /api/trained_models 获取的模型ID
    "image": "data:image/png;base64,iVBORw0KGgo..." // Base64 编码的图像数据
  }
  ```
- **响应 (JSON)**:
  ```json
  {
    "predicted_class": 7,
    "probabilities": [0.01, 0.01, 0.02, 0.05, 0.01, 0.1, 0.0, 0.8, 0.0, 0.0] 
  }
  ```

---

## 2. 核心数据结构

### 2.1 `TrainingHistoryEntry` (训练历史记录对象)
这是后端在训练完成后生成并保存到 `history.json` 的核心对象结构。

```javascript
{
  "job_id": "string", // 唯一任务ID
  "model_id": "string", // 如 "cnn_attention"
  "model_name": "string", // 如 "CNN + Attention"
  "status": "string", // "completed" 或 "error"
  "timestamp": "string", // ISO 8601 格式的完成时间
  "config": { // 训练配置
    "epochs": "number",
    "learning_rate": "number",
    "batch_size": "number"
  },
  "best_accuracy": "number", // 整个训练过程中验证集上的最佳准确率 (0-1)
  "final_train_loss": "number", // 最后一个 epoch 的训练集损失
  "final_val_loss": "number", // 最后一个 epoch 的验证集损失
  "samples_per_second": "number", // 整个训练过程的平均每秒处理样本数
  "duration_seconds": "number", // 总训练时长（秒）
  "epoch_metrics": [ // 每个 epoch 的详细数据
    {
      "epoch": "number",
      "val_loss": "number",
      "val_accuracy": "number",
      "train_loss": "number",
      "train_accuracy": "number",
      "epoch_duration_seconds": "number",
      "samples_per_second": "number"
    }
  ],
  "stability_metrics": { // 评估训练稳定性的指标
      "accuracy_variance": "number",
      "loss_variance": "number"
  },
  "environment_info": { // 训练环境信息
      "python_version": "string",
      "torch_version": "string",
      "device_name": "string" // 如 "CPU" 或 "NVIDIA GeForce RTX 3080"
  },
  "hyperparameters_extended": { // 更详细的超参数和模型结构信息
      "model_architecture": {
          "total_parameters": "number", // 模型总参数量
          "trainable_parameters": "number",
          "layer_info": "string" // 模型结构字符串
      }
  },
  "error_message": "string | null" // 如果训练失败，记录错误信息
}
```

### 2.2 `PredictionResult` (预测结果对象)
由 `/api/predict` 接口返回的对象。

```javascript
{
  "predicted_class": "number", // 预测的数字 (0-9)
  "probabilities": "array[number]" // 对应 0-9 每个数字的概率数组
}
```

---

## 3. 前端核心模块 (`frontend/js/`)

### 3.1 `api.js`
- **职责**: 封装所有与后端 API 的通信。所有 `fetch` 请求都应在此文件中定义，便于统一管理。
- **示例函数**: `fetchModels()`, `startTraining(modelIds, config)`, `fetchHistory()`, `predict(modelId, image)`.

### 3.2 `ui.js`
- **职责**: 负责所有与 DOM 操作和 UI 渲染相关的功能。它不处理业务逻辑，只根据传入的数据更新页面。
- **核心函数**:
    - `renderHistoryTable(history)`: 接收 `TrainingHistoryEntry[]` 数组，渲染历史记录表格。
    - `showDetailsModal(record)`: 接收单个 `TrainingHistoryEntry` 对象，渲染训练详情弹窗。
    - `renderPredictionResult(result)`: 接收 `PredictionResult` 对象，在界面上显示预测结果和概率图。
    - `createTrainingProgressBars(jobs)`: 创建并显示实时训练进度条。
- **重要依赖**: `showDetailsModal` 函数的 `record` 参数必须严格遵循 `TrainingHistoryEntry` 的结构。

### 3.3 `main.js`
- **职责**: 项目的启动器和协调器。
- **功能**:
    - 在页面加载时调用 `api.js` 中的函数获取初始数据 (如模型列表、历史记录)。
    - 将获取的数据传递给 `ui.js` 中的函数进行渲染。
    - 为页面上的按钮（如"开始训练"、"清空画布"）绑定事件监听器，并调用相应的功能模块。

### 3.4 `canvas.js`
- **职责**: 封装所有与 `<canvas>` 元素相关的操作。
- **功能**:
    - 初始化画布。
    - 处理鼠标/触摸的绘制事件。
    - 提供 `clearCanvas()` 和 `getImageData()` 等接口函数供 `main.js` 调用。

---
*文档最后更新时间: 2024-05-22* 