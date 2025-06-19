# MNIST 智能分析平台 - 后端启动脚本
# 自动激活虚拟环境并启动Flask后端服务

Write-Host "🚀 正在启动MNIST智能分析平台后端服务..." -ForegroundColor Green

# 检查是否在项目根目录
if (-not (Test-Path "backend/app.py")) {
    Write-Host "❌ 错误: 请在项目根目录 (mnist-platform) 下运行此脚本" -ForegroundColor Red
    Write-Host "当前目录: $(Get-Location)" -ForegroundColor Yellow
    Read-Host "按任意键退出"
    exit 1
}

# 检查虚拟环境
if (Test-Path "venv/Scripts/Activate.ps1") {
    Write-Host "📦 激活虚拟环境..." -ForegroundColor Yellow
    & "venv/Scripts/Activate.ps1"
} else {
    Write-Host "⚠️ 未找到虚拟环境，使用系统Python环境" -ForegroundColor Yellow
}

# 检查依赖
Write-Host "🔍 检查Python依赖..." -ForegroundColor Yellow
try {
    python -c "import flask, torch, torchvision, PIL; print('✅ 所有依赖已安装')"
} catch {
    Write-Host "❌ 缺少必要依赖，请运行: pip install -r requirements.txt" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

# 测试模型导入
Write-Host "🧠 测试模型创建..." -ForegroundColor Yellow
Set-Location backend
try {
    python -c "from models import get_model_instance; [get_model_instance(m) for m in ['mlp', 'cnn', 'rnn', 'mlp_attention', 'cnn_attention', 'rnn_attention']]; print('✅ 所有模型测试通过')"
    Write-Host "✅ 所有6个模型创建成功" -ForegroundColor Green
} catch {
    Write-Host "❌ 模型创建测试失败" -ForegroundColor Red
    Set-Location ..
    Read-Host "按任意键退出"
    exit 1
}

# 启动Flask服务
Write-Host "" 
Write-Host "🌟 启动Flask服务器..." -ForegroundColor Green
Write-Host "📡 服务地址: http://localhost:5000" -ForegroundColor Cyan
Write-Host "🎯 API地址: http://localhost:5000/api" -ForegroundColor Cyan
Write-Host ""
Write-Host "按 Ctrl+C 停止服务器" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Magenta

try {
    python app.py
} catch {
    Write-Host "❌ 服务启动失败" -ForegroundColor Red
} finally {
    Set-Location ..
    Write-Host ""
    Write-Host "👋 服务已停止" -ForegroundColor Yellow
    Read-Host "按任意键退出"
} 