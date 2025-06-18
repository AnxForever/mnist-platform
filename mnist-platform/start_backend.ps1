# MNIST智能分析平台 - 后端启动脚本
# PowerShell版本

Write-Host "🚀 启动MNIST智能分析平台后端服务器..." -ForegroundColor Green

# 检查虚拟环境
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "📁 激活虚拟环境..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "❌ 虚拟环境不存在，请先运行 python -m venv venv" -ForegroundColor Red
    exit 1
}

# 切换到backend目录并启动服务器
Write-Host "📂 切换到backend目录..." -ForegroundColor Yellow
Set-Location backend

Write-Host "🔥 启动Flask服务器..." -ForegroundColor Green
python app.py 