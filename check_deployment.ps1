# Render部署状态检查脚本 (PowerShell版本)

Write-Host "🔍 检查Render部署状态..." -ForegroundColor Blue

# 替换为你的实际Render URL
$RENDER_URL = "https://mnist-platform-backend.onrender.com"

Write-Host "🌐 测试API连接: $RENDER_URL/api/status" -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "$RENDER_URL/api/status" -Method GET -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ 后端服务运行正常" -ForegroundColor Green
        Write-Host "🎉 部署成功！" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 服务信息:" -ForegroundColor Cyan
        Write-Host "• 后端URL: $RENDER_URL" -ForegroundColor White
        Write-Host "• API状态: $RENDER_URL/api/status" -ForegroundColor White
        Write-Host "• 可用模型: $RENDER_URL/api/models" -ForegroundColor White
        Write-Host ""
        Write-Host "🔗 现在可以更新前端配置，连接到后端服务" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  后端服务响应异常 (状态码: $($response.StatusCode))" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ 后端服务无法访问" -ForegroundColor Red
    Write-Host "请检查Render控制台的部署日志" -ForegroundColor Yellow
    Write-Host "错误详情: $($_.Exception.Message)" -ForegroundColor Red
} 