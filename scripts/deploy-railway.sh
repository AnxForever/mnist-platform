#!/bin/bash

# MNIST智能分析平台 - Railway部署脚本
# 用法: ./scripts/deploy-railway.sh

echo "🚀 开始部署到Railway平台..."

# 检查必要文件
if [ ! -f "deployment/railway.toml" ]; then
    echo "❌ 错误: deployment/railway.toml 文件不存在"
    exit 1
fi

if [ ! -f "backend/requirements.txt" ]; then
    echo "❌ 错误: backend/requirements.txt 文件不存在"
    exit 1
fi

# 复制配置文件到根目录 (Railway平台需要)
echo "📁 准备部署配置文件..."
cp deployment/railway.toml railway.toml
cp deployment/railway.json railway.json 2>/dev/null || echo "ℹ️  railway.json 不存在，跳过"

# 提交到Git
echo "📤 提交代码到Git仓库..."
git add .
git commit -m "Deploy: Update configuration for Railway deployment"
git push origin main

echo "✅ 代码已推送到GitHub"
echo "🌐 请访问Railway控制台查看部署状态"
echo "📖 部署完成后，你的应用将在以下地址可用:"
echo "   https://your-app-name.railway.app"

# 清理临时文件
rm -f railway.toml railway.json

echo "🎉 部署脚本执行完成！" 