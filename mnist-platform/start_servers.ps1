# === MNIST 智能分析平台 一键启动脚本 ===
# 作者: 你的AI助手
# 功能: 此脚本会自动激活Python虚拟环境，并同时启动后端Flask服务和前端HTTP服务。
# ==========================================

# 设置一个醒目的标题
Write-Host "=== MNIST 智能分析平台启动脚本 ===" -ForegroundColor Green

# --- 1. 设置工作目录 ---
# 获取脚本所在的目录，并将其设置为当前工作目录
# 这样做可以确保后续所有相对路径都能正确找到文件
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
Write-Host "项目目录: $ProjectRoot" -ForegroundColor Yellow

# --- 2. 激活 Python 虚拟环境 ---
# 虚拟环境就像一个独立的工具箱，里面有我们项目需要的所有特定版本的工具（库）
$VenvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host "正在激活虚拟环境..." -ForegroundColor Cyan
    & $VenvPath
} else {
    Write-Host "错误：在 '$VenvPath' 未找到虚拟环境！" -ForegroundColor Red
    exit
}

# --- 3. 启动后端 Flask 服务器 ---
# -NoExit 参数让新的 PowerShell 窗口在命令执行后保持打开，方便我们看日志
# -Command 参数后面跟着要执行的一系列命令
Write-Host "正在启动后端服务器 (端口 5000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectRoot'; & $VenvPath; Write-Host '--- 后端服务已启动 (app.py) ---' -ForegroundColor Green; python backend/app.py"
)

# 短暂等待，确保后端服务有足够的时间开始初始化
Start-Sleep -Seconds 3

# --- 4. 启动前端 HTTP 服务器 ---
# 这个服务器负责将 index.html 和相关的 js/css 文件发送给你的浏览器
Write-Host "正在启动前端服务器 (端口 8080)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", 
    "-Command",
    "cd '$ProjectRoot'; & $VenvPath; Write-Host '--- 前端服务已启动 (serve.py) ---' -ForegroundColor Green; python serve.py"
)

# --- 5. 自动打开浏览器 ---
Write-Host "等待所有服务稳定..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
Write-Host "正在浏览器中打开应用..." -ForegroundColor Green
Start-Process "http://localhost:8080"

# --- 脚本结束语 ---
Write-Host "" 
Write-Host "=== ✅ 所有服务均已启动 ===" -ForegroundColor Green
Write-Host "前端访问地址: http://localhost:8080" -ForegroundColor White
Write-Host "后端API地址: http://localhost:5000" -ForegroundColor White
Write-Host ""
Write-Host "你可以关闭此窗口，服务会继续在后台运行。"
Write-Host "如需停止服务，请手动关闭新打开的两个PowerShell窗口。"
Read-Host -Prompt "按 Enter 键退出此启动向导" 