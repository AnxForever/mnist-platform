#!/usr/bin/env python3
"""
MNIST智能分析平台 - 主入口文件
适配Render云端部署
"""

import os
import sys

# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# 导入backend中的Flask应用
from backend.app import app

if __name__ == '__main__':
    # 获取端口号，默认5000，支持Render的PORT环境变量
    port = int(os.environ.get('PORT', 5000))
    
    # 启动Flask应用
    print(f"🚀 启动MNIST智能分析平台，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 