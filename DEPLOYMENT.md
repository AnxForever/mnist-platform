# 🚀 MNIST智能分析平台 - 云端部署指南

## 🚀 快速部署概览

本项目支持多平台云端部署，推荐使用 **Render** 平台进行0成本部署。

### 🏗️ 项目架构

```
前端 (Vercel) ←→ 后端 (Render)
    ↓                ↓
静态托管         Flask + PyTorch
```

## 📋 部署前准备

### 1. 环境要求
- Python 3.13+
- Git 版本控制
- GitHub 账户
- 网络连接

### 2. 依赖版本 (已优化)
```
Flask==2.3.3
torch==2.6.0+cpu  # CPU专用版本，减少包大小
torchvision==0.17.0+cpu
numpy==1.26.4
Python==3.13.4    # 完全兼容
```

## 🎯 推荐方案：Render部署 (0成本)

### 方式一：自动化脚本部署

```bash
# 1. 执行自动化部署准备
./scripts/deploy-render.sh

# 2. 按照脚本输出的指示在Render网站完成部署

# 3. 验证部署状态
./check_deployment.sh
```

### 方式二：网站手动操作 (推荐)

#### 第一步：访问Render

1. 打开 [render.com](https://render.com)
2. 使用GitHub账户登录

#### 第二步：创建Web Service

1. 点击 **"Create a new Web Service"**
2. 连接GitHub仓库：选择 `mnist-platform`
3. 如果仓库未显示，点击 "Configure GitHub App" 授权

#### 第三步：配置部署设置

| 配置项 | 值 |
|--------|-----|
| **Name** | `mnist-platform-backend` |
| **Root Directory** | `backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir` |
| **Start Command** | `python app.py` |

#### 第四步：环境变量配置

添加以下环境变量：

| 变量名 | 值 | 说明 |
|--------|-----|------|
| `PORT` | `10000` | 服务端口 |
| `FLASK_ENV` | `production` | 生产环境 |
| `PYTHONUNBUFFERED` | `1` | 实时日志输出 |
| `FLASK_SKIP_DOTENV` | `1` | 跳过dotenv加载 |

#### 第五步：选择计划

- 选择 **"Free"** 计划 (0成本)
- 限制：512MB RAM, 自动休眠

#### 第六步：创建服务

1. 点击 **"Create Web Service"**
2. 等待部署完成 (首次约5-10分钟)

#### 第七步：监控部署

查看部署日志，确认以下输出：
```
🚀 启动MNIST智能分析平台后端服务
📡 监听地址: 0.0.0.0:10000
🔧 环境模式: 生产
🧠 PyTorch设备: CPU
✅ 后端服务启动完成，等待前端连接...
```

## 🔧 高级配置

### 性能优化配置

1. **CPU专用PyTorch**: 使用 `torch==2.6.0+cpu` 减少包大小50%
2. **内存优化**: 启用 `--no-cache-dir` 安装选项
3. **缓存设置**: 静态文件1年缓存期
4. **健康检查**: 自动监控 `/api/status` 端点

### 自定义域名 (可选)

1. 在Render控制台进入服务设置
2. 添加自定义域名
3. 配置DNS CNAME记录指向Render URL

### 环境变量详解

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `PORT` | HTTP服务端口 | `5000` |
| `HOST` | 监听地址 | `0.0.0.0` |
| `FLASK_ENV` | Flask环境模式 | `development` |
| `PYTHONUNBUFFERED` | Python输出缓冲 | 未设置 |
| `FLASK_SKIP_DOTENV` | 跳过.env文件 | 未设置 |

## 🧪 部署验证

### API测试端点

```bash
# 基础健康检查
curl https://your-app.onrender.com/api/status

# 获取可用模型
curl https://your-app.onrender.com/api/models

# 获取预训练模型
curl https://your-app.onrender.com/api/pretrained_models
```

### 预期响应

健康检查应返回：
```json
{
  "status": "running",
  "timestamp": "2024-xx-xx xx:xx:xx",
  "environment": "production",
  "pytorch_device": "cpu"
}
```

## 🔗 前端连接配置

部署完成后，需要更新前端配置：

1. 编辑 `frontend/js/config.js`
2. 更新 `API_BASE_URL` 为Render URL：
   ```javascript
   const API_BASE_URL = 'https://mnist-platform-backend.onrender.com';
   ```

## 🐛 故障排除

### 常见问题

#### 1. 部署超时
- **原因**: 依赖安装时间过长
- **解决**: 使用CPU版PyTorch，减少包大小

#### 2. 内存超限
- **原因**: Free计划512MB限制
- **解决**: 优化代码，减少内存使用

#### 3. 冷启动延迟
- **原因**: Free计划自动休眠
- **解决**: 使用付费计划或预热请求

#### 4. Python版本错误
- **原因**: 版本不兼容
- **解决**: 检查 `deployment/runtime.txt` 文件

### 调试步骤

1. **查看构建日志**
   - 在Render控制台检查Build Logs
   - 确认所有依赖安装成功

2. **检查运行日志**
   - 查看Service Logs
   - 确认Flask应用启动成功

3. **测试API连通性**
   ```bash
   curl -I https://your-app.onrender.com/api/status
   ```

4. **本地测试**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

## 📊 监控和维护

### 性能监控

- **Render内置监控**: CPU、内存、响应时间
- **自定义健康检查**: `/api/status` 端点
- **日志聚合**: 通过Render控制台查看

### 更新部署

```bash
# 推送代码更新
git add .
git commit -m "update: 功能更新"
git push origin main

# Render自动重新部署
```

### 备份策略

- **代码备份**: GitHub自动版本控制
- **模型备份**: 使用云存储服务 (可选)
- **配置备份**: 导出环境变量设置

## 🌟 最佳实践

1. **分支策略**: 使用main分支进行生产部署
2. **环境隔离**: 开发/测试/生产环境分离
3. **监控告警**: 设置关键API的监控
4. **安全考虑**: 定期更新依赖版本
5. **性能优化**: 使用CDN加速静态资源

## 📞 技术支持

- **文档问题**: 查看项目README.md
- **代码问题**: 提交GitHub Issue
- **部署问题**: 检查Render官方文档
- **性能问题**: 考虑升级到付费计划

---

✅ **部署完成后，你将拥有一个完全云端的MNIST深度学习平台！**