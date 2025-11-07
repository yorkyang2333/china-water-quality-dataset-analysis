#!/usr/bin/env python
"""
水质监测可视化平台 - 启动脚本
简单的一键启动方式
"""

import sys
import os

# 获取项目根目录
project_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(project_dir, 'backend')

# 将项目根目录添加到路径
sys.path.insert(0, project_dir)

print("="*60)
print("  Water Quality Monitoring Platform")
print("="*60)
print(f"Project directory: {project_dir}")
print(f"Backend directory: {backend_dir}")
print()

# 导入并启动应用
from backend.app import app

if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60)
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)
