#!/usr/bin/env python3
"""
简单的HTTP服务器，用于服务前端文件
"""
import http.server
import socketserver
import os
import webbrowser
from threading import Timer

PORT = 8080
DIRECTORY = "frontend"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def open_browser():
    """延迟2秒后自动打开浏览器"""
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == "__main__":
    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"前端服务器启动在端口 {PORT}")
        print(f"访问地址: http://localhost:{PORT}")
        print(f"前端文件目录: {os.path.abspath(DIRECTORY)}")
        print("按 Ctrl+C 停止服务器")
        
        # 2秒后自动打开浏览器
        Timer(2.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止") 