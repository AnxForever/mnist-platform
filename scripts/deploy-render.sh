#!/bin/bash

# MNIST平台 - Render自动化部署脚本
# 作者: AI助手
# 功能: 自动将MNIST深度学习平台部署到Render云服务

set -e  # 遇到错误立即退出

echo "🚀 开始MNIST平台Render自动化部署..."
echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 1. 环境检查
echo "🔍 检查部署环境..."

if ! command -v git &> /dev/null; then
    echo "❌ 错误: 未找到git命令，请先安装Git"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "❌ 错误: 未找到curl命令，请先安装curl"
    exit 1
fi

# 检查是否在git仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ 错误: 当前目录不是Git仓库"
    exit 1
fi

echo "✅ 环境检查通过"

# 2. 代码提交和推送
echo "📦 准备代码推送..."

# 检查是否有未提交的更改
if [[ -n $(git status --porcelain) ]]; then
    echo "📝 发现未提交的更改，正在提交..."
    git add .
    git commit -m "deploy: Render部署优化 - $(date '+%Y%m%d_%H%M%S')"
else
    echo "✅ 代码已是最新状态"
fi

echo "🔄 推送代码到远程仓库..."
if git push origin main; then
    echo "✅ 代码推送成功"
else
    echo "❌ 代码推送失败，请检查网络连接和权限"
    exit 1
fi

# 3. 部署配置验证
echo "🔧 验证部署配置..."

if [[ ! -f "deployment/render.yaml" ]]; then
    echo "❌ 错误: 缺少render.yaml配置文件"
    exit 1
fi

if [[ ! -f "backend/requirements.txt" ]]; then
    echo "❌ 错误: 缺少requirements.txt依赖文件"
    exit 1
fi

if [[ ! -f "deployment/runtime.txt" ]]; then
    echo "❌ 错误: 缺少runtime.txt版本文件"
    exit 1
fi

echo "✅ 部署配置验证通过"

# 4. 显示部署信息
echo ""
echo "📋 部署配置摘要:"
echo "----------------------------------------"
echo "🐍 Python版本: $(cat deployment/runtime.txt)"
echo "🌐 服务名称: mnist-platform-backend"
echo "📁 根目录: backend"
echo "🔗 健康检查: /api/status"
echo "💰 计划: Free"
echo ""

# 5. 部署指引
echo "🎯 接下来的手动操作步骤:"
echo "=========================================="
echo "1. 访问 https://render.com 并登录GitHub账户"
echo "2. 点击 'Create a new Web Service'"
echo "3. 选择你的 'mnist-platform' 仓库"
echo "4. 配置以下设置:"
echo "   • Name: mnist-platform-backend"
echo "   • Root Directory: backend"
echo "   • Runtime: Python 3"
echo "   • Build Command: pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir"
echo "   • Start Command: python app.py"
echo "5. 设置环境变量:"
echo "   • PORT = 10000"
echo "   • FLASK_ENV = production"
echo "   • PYTHONUNBUFFERED = 1"
echo "   • FLASK_SKIP_DOTENV = 1"
echo "6. 选择 'Free' 计划"
echo "7. 点击 'Create Web Service'"
echo ""

# 6. 部署状态检查脚本
echo "🔍 生成部署状态检查脚本..."
cat > check_deployment.sh << 'EOF'
#!/bin/bash
# Render部署状态检查脚本

echo "🔍 检查Render部署状态..."

# 替换为你的实际Render URL
RENDER_URL="https://mnist-platform-backend.onrender.com"

echo "🌐 测试API连接: $RENDER_URL/api/status"

if curl -f -s "$RENDER_URL/api/status" > /dev/null; then
    echo "✅ 后端服务运行正常"
    echo "🎉 部署成功！"
    echo ""
    echo "📋 服务信息:"
    echo "• 后端URL: $RENDER_URL"
    echo "• API状态: $RENDER_URL/api/status"
    echo "• 可用模型: $RENDER_URL/api/models"
    echo ""
    echo "🔗 现在可以更新前端配置，连接到后端服务"
else
    echo "❌ 后端服务无法访问"
    echo "请检查Render控制台的部署日志"
fi
EOF

chmod +x check_deployment.sh
echo "✅ 状态检查脚本已生成: ./check_deployment.sh"

echo ""
echo "🎉 自动化部署准备完成！"
echo "📝 请按照上述步骤在Render网站完成部署"
echo "🕐 首次部署通常需要5-10分钟"
echo "✅ 部署完成后运行 ./check_deployment.sh 验证服务状态"
echo "==========================================" 