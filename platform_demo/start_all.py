#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动脚本 - 同时启动Flask后端和Vue前端
"""

import os
import sys
import subprocess
import time
import signal
import webbrowser
from pathlib import Path

# 设置Windows控制台UTF-8编码支持
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 全局进程列表，用于清理
processes = []

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n\n🛑 接收到停止信号，正在关闭服务...")
    cleanup()
    sys.exit(0)

def cleanup():
    """清理所有子进程"""
    for name, process in processes:
        if process and process.poll() is None:
            print(f"⏹️  停止 {name}...")
            try:
                if sys.platform == 'win32':
                    # Windows下需要用taskkill来结束进程树
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                   capture_output=True)
                else:
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                print(f"   警告: 停止 {name} 时出错: {e}")

def check_node_npm():
    """检查Node.js和npm是否安装"""
    try:
        node_version = subprocess.run(
            ['node', '--version'],
            capture_output=True,
            text=True,
            shell=True
        )
        npm_version = subprocess.run(
            ['npm', '--version'],
            capture_output=True,
            text=True,
            shell=True
        )

        if node_version.returncode == 0 and npm_version.returncode == 0:
            print(f"✅ Node.js: {node_version.stdout.strip()}")
            print(f"✅ npm: {npm_version.stdout.strip()}")
            return True
        else:
            print("❌ Node.js或npm未安装")
            return False
    except Exception as e:
        print(f"❌ Node.js或npm检测失败: {e}")
        print("   请从 https://nodejs.org 下载安装")
        return False

def check_python():
    """检查Python环境"""
    print(f"✅ Python: {sys.version.split()[0]}")
    return True

def install_vue_dependencies():
    """安装Vue依赖"""
    vue_dir = Path("frontend-vue")
    node_modules = vue_dir / "node_modules"

    if not node_modules.exists():
        print("📥 安装Vue项目依赖（这可能需要几分钟）...")
        try:
            result = subprocess.run(
                ['npm', 'install'],
                cwd=vue_dir,
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode != 0:
                print(f"❌ 依赖安装失败: {result.stderr}")
                return False
            print("✅ Vue依赖安装成功")
        except Exception as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
    else:
        print("✅ Vue依赖已安装")

    return True

def start_backend():
    """启动Flask后端"""
    print("\n🚀 启动Flask后端...")
    print("   后端地址: http://localhost:5000")

    try:
        # 查找run.py的位置
        if Path("run.py").exists():
            run_script = "run.py"
        elif Path("../run.py").exists():
            run_script = "../run.py"
        else:
            print("❌ 找不到run.py文件")
            return None

        process = subprocess.Popen(
            [sys.executable, run_script],
            shell=True
        )

        # 等待后端启动
        print("   等待后端启动...")
        time.sleep(5)  # 增加等待时间以确保服务有足够时间初始化

        if process.poll() is None:
            print("✅ Flask后端已启动（日志将直接显示在下方）")
            return process
        else:
            print("❌ Flask后端启动失败，请检查上方日志")
            return None

    except Exception as e:
        print(f"❌ 启动后端失败: {e}")
        return None

def start_frontend():
    """启动Vue前端"""
    print("\n🚀 启动Vue前端...")
    print("   前端地址: http://localhost:3000")

    vue_dir = Path("frontend-vue")

    try:
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=vue_dir,
            shell=True
        )

        # 等待前端启动
        print("   等待前端启动...")
        time.sleep(10) # Vue启动可能较慢，增加等待时间

        if process.poll() is None:
            print("✅ Vue前端已启动（日志将直接显示在下方）")
            return process
        else:
            print("❌ Vue前端启动失败，请检查上方日志")
            return None

    except Exception as e:
        print(f"❌ 启动前端失败: {e}")
        return None

def open_browser():
    """打开浏览器"""
    print("\n🌐 正在打开浏览器...")
    time.sleep(2)
    try:
        webbrowser.open('http://localhost:3000')
        print("✅ 浏览器已打开")
    except Exception as e:
        print(f"⚠️  自动打开浏览器失败: {e}")
        print("   请手动访问: http://localhost:3000")

def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("🎉 水质监测平台 - 一键启动")
    print("=" * 60)

def print_success():
    """打印成功信息"""
    print("\n" + "=" * 60)
    print("✨ 所有服务已成功启动!")
    print("=" * 60)
    print("\n📋 服务地址:")
    print("   🔧 后端API: http://localhost:5000")
    print("   🎨 前端界面: http://localhost:3000")
    print("\n💡 提示:")
    print("   - 前端会自动代理API请求到后端")
    print("   - 修改代码会自动热重载")
    print("   - 按 Ctrl+C 停止所有服务")
    print("=" * 60)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)

    print_welcome()

    # 检查环境
    print("\n🔍 检查运行环境...")
    if not check_python():
        return False

    if not check_node_npm():
        return False

    # 安装Vue依赖
    if not install_vue_dependencies():
        return False

    print("\n" + "=" * 60)

    # 启动后端
    backend_process = start_backend()
    if backend_process:
        processes.append(("Flask后端", backend_process))
    else:
        print("\n⚠️  后端启动失败，但可以继续启动前端")

    # 启动前端
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ 前端启动失败")
        cleanup()
        return False

    processes.append(("Vue前端", frontend_process))

    # 打印成功信息
    print_success()

    # 打开浏览器
    open_browser()

    # 保持运行
    try:
        print("\n📡 服务运行中...")
        print("   (按 Ctrl+C 停止所有服务)\n")

        while True:
            time.sleep(1)

            # 检查进程是否还在运行
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n⚠️  {name} 已停止")
                    cleanup()
                    return False

    except KeyboardInterrupt:
        pass

    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ 启动失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        cleanup()
        sys.exit(1)
    finally:
        cleanup()
        print("\n👋 已停止所有服务")
