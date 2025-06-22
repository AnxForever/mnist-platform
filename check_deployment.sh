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