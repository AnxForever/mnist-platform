# 🚀 后端部署配置目录

这个目录包含了MNIST智能分析平台后端服务的所有部署配置文件。

## 📁 文件说明

### 平台配置文件

- **`render.yaml`** - Render平台部署配置
  - 工作目录: `backend/`
  - 启动命令: `python app.py`
  - 端口: 10000

- **`railway.toml`** + **`railway.json`** - Railway平台部署配置
  - 支持自动部署和环境变量管理

- **`Procfile`** - Heroku风格的进程配置文件
  - 兼容多种云平台

- **`runtime.txt`** - Python版本指定
  - 确保使用Python 3.13.4

## 🛠️ 使用方法

### 手动部署

1. **Render部署**:
   ```bash
   cp deployment/render.yaml ./render.yaml
   git add . && git commit -m "Deploy to Render" && git push
   ```

2. **Railway部署**:
   ```bash
   cp deployment/railway.toml ./railway.toml
   git add . && git commit -m "Deploy to Railway" && git push
   ```

### 自动化部署

使用项目根目录下的部署脚本：

```bash
# 部署到Render
./scripts/deploy-render.sh

# 部署到Railway  
./scripts/deploy-railway.sh
```

## 🏗️ 架构说明

- **工作目录**: 所有平台均使用 `backend/` 作为工作目录
- **依赖管理**: 统一使用 `backend/requirements.txt`
- **启动入口**: `backend/app.py`
- **环境变量**: 
  - `PORT`: 服务端口（默认5000）
  - `FLASK_ENV`: 运行环境（development/production）
  - `HOST`: 监听地址（默认0.0.0.0）

## 🔧 配置原则

1. **单一真相源**: 每种配置只有一个权威文件
2. **环境无关**: 代码在任何云平台都能正常运行
3. **标准化**: 遵循各平台的最佳实践
4. **简化维护**: 最小化配置文件数量和复杂度

## 📝 注意事项

- 前端部署配置（`vercel.json`）保持在项目根目录
- 不要直接修改根目录的配置文件，统一在此目录管理
- 部署前确保 `backend/requirements.txt` 包含所有必要依赖 