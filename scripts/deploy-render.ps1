# MNIST平台 - Render自动化部署脚本 (Windows PowerShell版本)
# 作者: AI助手
# 功能: 自动将MNIST深度学习平台部署到Render云服务

$ErrorActionPreference = "Stop"

Write-Host "🚀 开始MNIST平台Render自动化部署..." -ForegroundColor Green
Write-Host "⏰ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Yellow

# 1. 环境检查
Write-Host "🔍 检查部署环境..." -ForegroundColor Blue

# 检查Git命令
try {
    git --version | Out-Null
    Write-Host "✅ Git 已安装" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 未找到git命令，请先安装Git" -ForegroundColor Red
    exit 1
}

# 检查curl命令
try {
    curl --version | Out-Null
    Write-Host "✅ Curl 已安装" -ForegroundColor Green
} catch {
    Write-Host "⚠️  警告: 未找到curl命令，将使用PowerShell进行网络测试" -ForegroundColor Yellow
}

# 检查是否在git仓库中
try {
    git rev-parse --git-dir | Out-Null
    Write-Host "✅ Git仓库检查通过" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 当前目录不是Git仓库" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 环境检查通过" -ForegroundColor Green

# 2. 代码提交和推送
Write-Host "📦 准备代码推送..." -ForegroundColor Blue

# 检查是否有未提交的更改
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "📝 发现未提交的更改，正在提交..." -ForegroundColor Yellow
    git add .
    $commitMessage = "deploy: Render部署优化 - $(Get-Date -Format 'yyyyMMdd_HHmmss')"
    git commit -m $commitMessage
    Write-Host "✅ 代码已提交" -ForegroundColor Green
} else {
    Write-Host "✅ 代码已是最新状态" -ForegroundColor Green
}

Write-Host "🔄 推送代码到远程仓库..." -ForegroundColor Blue
try {
    git push origin main
    Write-Host "✅ 代码推送成功" -ForegroundColor Green
} catch {
    Write-Host "❌ 代码推送失败，请检查网络连接和权限" -ForegroundColor Red
    Write-Host "错误详情: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 3. 部署配置验证
Write-Host "🔧 验证部署配置..." -ForegroundColor Blue

$configFiles = @(
    "deployment/render.yaml",
    "backend/requirements.txt", 
    "deployment/runtime.txt"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file 存在" -ForegroundColor Green
    } else {
        Write-Host "❌ 错误: 缺少 $file 配置文件" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ 部署配置验证通过" -ForegroundColor Green

# 4. 显示部署信息
Write-Host ""
Write-Host "📋 部署配置摘要:" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Yellow
$pythonVersion = Get-Content "deployment/runtime.txt" -Raw
Write-Host "🐍 Python版本: $($pythonVersion.Trim())" -ForegroundColor White
Write-Host "🌐 服务名称: mnist-platform-backend" -ForegroundColor White
Write-Host "📁 根目录: backend" -ForegroundColor White
Write-Host "🔗 健康检查: /api/status" -ForegroundColor White
Write-Host "💰 计划: Free" -ForegroundColor White
Write-Host ""

# 5. 部署指引
Write-Host "🎯 接下来的手动操作步骤:" -ForegroundColor Magenta
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "1. 访问 https://render.com 并登录GitHub账户" -ForegroundColor White
Write-Host "2. 点击 'Create a new Web Service'" -ForegroundColor White
Write-Host "3. 选择你的 'mnist-platform' 仓库" -ForegroundColor White
Write-Host "4. 配置以下设置:" -ForegroundColor White
Write-Host "   • Name: mnist-platform-backend" -ForegroundColor Gray
Write-Host "   • Root Directory: backend" -ForegroundColor Gray
Write-Host "   • Runtime: Python 3" -ForegroundColor Gray
$buildCommand = "pip install --upgrade pip; pip install -r requirements.txt --no-cache-dir"
Write-Host "   • Build Command: $buildCommand" -ForegroundColor Gray
Write-Host "   • Start Command: python app.py" -ForegroundColor Gray
Write-Host "5. 设置环境变量:" -ForegroundColor White
Write-Host "   • PORT = 10000" -ForegroundColor Gray
Write-Host "   • FLASK_ENV = production" -ForegroundColor Gray
Write-Host "   • PYTHONUNBUFFERED = 1" -ForegroundColor Gray
Write-Host "   • FLASK_SKIP_DOTENV = 1" -ForegroundColor Gray
Write-Host "6. 选择 'Free' 计划" -ForegroundColor White
Write-Host "7. 点击 'Create Web Service'" -ForegroundColor White
Write-Host ""

# 6. 生成部署状态检查脚本
Write-Host "🔍 生成部署状态检查脚本..." -ForegroundColor Blue

$checkScriptContent = @"
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
"@

$checkScriptContent | Out-File -FilePath "check_deployment.ps1" -Encoding UTF8

Write-Host "✅ Windows状态检查脚本已生成: check_deployment.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 自动化部署准备完成！" -ForegroundColor Green
Write-Host "📝 请按照上述步骤在Render网站完成部署" -ForegroundColor Cyan
Write-Host "🕐 首次部署通常需要5-10分钟" -ForegroundColor Yellow
Write-Host "✅ 部署完成后运行 .\check_deployment.ps1 验证服务状态" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Yellow

# 询问是否直接打开Render网站
Write-Host ""
$openSite = Read-Host "🌐 是否现在打开Render网站开始部署？(Y/N)"
if ($openSite -eq "Y" -or $openSite -eq "y") {
    Start-Process "https://render.com"
    Write-Host "✅ 已打开Render网站，请按照上述步骤完成部署" -ForegroundColor Green
} 