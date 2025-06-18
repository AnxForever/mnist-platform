# 🧠 MNIST 智能分析平台

> 一个基于深度学习的MNIST数字分类系统，支持6种不同的神经网络模型训练与对比分析

## ✨ 功能特性

### 🔥 六大模型军团
- **基础模型（3个）**：MLP、CNN、RNN
- **Attention增强版（3个）**：MLP+Attention、CNN+Attention、RNN+Attention

### 🚀 核心功能
- ⚡ **并发训练**：最多支持3个模型同时训练
- 📊 **实时监控**：训练进度、准确率、损失值实时显示
- 💾 **智能存档**：自动保存最佳模型检查点，防止训练成果丢失
- 🎨 **手写识别**：在线画板 + 模型预测
- 📈 **性能对比**：多维度图表分析不同模型表现
- 📚 **历史记录**：完整的训练历史管理

### 🛡️ 可靠性保障
- **线程安全**：并发训练时的文件写操作保护
- **错误恢复**：训练失败时的详细错误信息记录
- **检查点机制**：训练过程中自动保存最佳状态
- **网络重试**：前端API请求的自动重试机制

## 🏗️ 技术架构

### 后端技术栈
- **框架**：Flask + Flask-CORS
- **深度学习**：PyTorch + torchvision
- **并发处理**：concurrent.futures.ThreadPoolExecutor
- **数据存储**：JSON文件 + PyTorch模型文件

### 前端技术栈
- **基础**：原生 HTML5 + CSS3 + JavaScript (ES6+)
- **图表**：Chart.js
- **架构**：模块化设计（API、UI、主逻辑分离）

## 📁 项目结构

```
mnist-platform/
├── backend/                    # 后端代码
│   ├── models/                 # 深度学习模型定义
│   │   ├── base_model.py       # 模型基类（必须）
│   │   ├── mlp.py             # MLP基础模型
│   │   ├── cnn.py             # CNN基础模型
│   │   ├── rnn.py             # RNN基础模型
│   │   ├── mlp_attention.py   # MLP + Attention
│   │   ├── cnn_attention.py   # CNN + Attention
│   │   ├── rnn_attention.py   # RNN + Attention
│   │   └── attention_layers.py # 通用注意力模块
│   ├── core/                  # 核心业务逻辑
│   │   ├── training_manager.py # 并发训练管理
│   │   └── persistence.py     # 数据持久化
│   ├── data/                  # 数据处理
│   │   └── loader.py          # MNIST数据加载器
│   ├── saved_models/          # 训练完成的模型文件
│   ├── checkpoints/           # 训练过程中的检查点
│   └── app.py                 # Flask应用入口
├── frontend/                  # 前端代码
│   ├── css/
│   │   └── style.css          # 全局样式
│   ├── js/
│   │   ├── main.js            # 主应用逻辑
│   │   ├── api.js             # API请求封装
│   │   └── ui.js              # DOM操作
│   └── index.html             # 主页面
├── requirements.txt           # Python依赖
├── .gitignore                # Git忽略文件
└── README.md                 # 项目说明
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd mnist-platform

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 启动后端服务
```bash
cd backend
python app.py
```
后端服务将在 `http://localhost:5000` 启动

### 3. 访问前端界面
使用浏览器打开 `frontend/index.html`，或通过HTTP服务器访问：
```bash
# 使用Python内置服务器
cd frontend
python -m http.server 8080
```
然后访问 `http://localhost:8080`

## 🎯 使用指南

### 模型训练
1. 在"模型训练"页面选择要训练的模型
2. 配置训练参数（训练轮数、学习率）
3. 点击"开始训练"，系统支持同时训练多个模型
4. 实时查看训练进度和指标

### 手写识别
1. 切换到"手写识别"页面
2. 在画板上绘制数字
3. 选择已训练的模型
4. 点击"识别"查看预测结果

### 性能分析
1. "训练结果"页面查看所有历史记录
2. "模型对比"页面进行多维度性能分析

## 🔧 开发注意事项

### 关键约定
1. **所有模型必须继承 `BaseModel`** - 确保接口一致性
2. **所有文件写操作必须加锁** - 防止并发冲突
3. **训练失败必须记录完整错误信息** - 便于问题定位
4. **每次最佳准确率提升必须保存检查点** - 防止训练成果丢失

### 持久化策略
- **最终存档**：训练成功后保存最终模型和历史记录
- **最佳检查点**：训练过程中实时保存最佳状态

## 🗺️ 开发路线图

- [x] 第一阶段：后端基础架构和核心模型
- [ ] 第二阶段：Attention模型与并发系统
- [ ] 第三阶段：前端界面与核心交互
- [ ] 第四阶段：前端功能补全
- [ ] 第五阶段：联调测试与优化
- [ ] 第六阶段：高级功能（雷达图、混淆矩阵等）

## 🌟 未来规划

### 可视化增强
- 雷达图多维度模型对比
- 混淆矩阵热力图
- 学习曲线动画效果

### 功能扩展
- 模型导出/导入功能
- 超参数智能推荐
- 分布式训练支持

### 部署优化
- Vercel无服务器部署适配
- 云存储集成（Vercel Blob / AWS S3）
- 第三方训练服务集成

## 📝 开发文档

详细的技术设计文档请参考项目根目录的 `项目文档.md` 文件，包含：
- 完整的API接口规格
- 数据模型设计
- 前端组件设计
- 部署方案说明

---

**本项目是开发学习的完整技术实践，展示了从零开始构建一个深度学习Web应用的全过程。** 