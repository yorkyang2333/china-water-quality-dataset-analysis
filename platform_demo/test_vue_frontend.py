#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vue.js 前端测试启动脚本
测试现代化Vue.js前端与现有后端的兼容性
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# 设置Windows控制台UTF-8编码支持
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_node_npm():
    """检查Node.js和npm是否安装"""
    try:
        # Windows需要shell=True来正确执行命令
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
            print(f"   Node返回码: {node_version.returncode}")
            print(f"   npm返回码: {npm_version.returncode}")
            return False
    except FileNotFoundError as e:
        print("❌ Node.js或npm未安装")
        print("请从 https://nodejs.org 下载安装")
        print(f"   错误详情: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查Node.js/npm时发生错误: {e}")
        return False

def setup_vue_project():
    """设置Vue项目"""
    vue_dir = Path("frontend-vue")
    
    if not vue_dir.exists():
        print("❌ Vue项目目录不存在")
        return False
    
    print("📦 检查Vue项目依赖...")
    
    # 检查node_modules是否存在
    node_modules = vue_dir / "node_modules"
    if not node_modules.exists():
        print("📥 安装Vue项目依赖...")
        try:
            os.chdir(vue_dir)
            result = subprocess.run(
                ['npm', 'install'],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode != 0:
                print(f"❌ 依赖安装失败: {result.stderr}")
                return False
            print("✅ 依赖安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
        finally:
            os.chdir("..")
    else:
        print("✅ 依赖已安装")
    
    return True

def check_backend():
    """检查后端服务是否运行"""
    import requests
    
    try:
        response = requests.get('http://localhost:5000/api/stations', timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务运行正常")
            return True
        else:
            print(f"⚠️ 后端服务响应异常: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ 后端服务未运行")
        print("请先启动后端服务: python run.py")
        return False

def start_vue_dev_server():
    """启动Vue开发服务器"""
    vue_dir = Path("frontend-vue")
    
    print("\n🚀 启动Vue开发服务器...")
    print("=" * 50)
    print("Vue.js 开发服务器将在 http://localhost:3000 启动")
    print("前端将通过代理访问 http://localhost:5000 的后端API")
    print("=" * 50)
    
    try:
        os.chdir(vue_dir)
        
        # 启动Vue开发服务器
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            shell=True
        )
        
        print("\n📡 正在启动服务器...")
        
        # 等待服务器启动
        startup_timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < startup_timeout:
            if process.poll() is not None:
                print("❌ Vue服务器启动失败")
                print(process.stdout.read())
                return False
            
            # 检查是否有输出
            try:
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                
                # 检查服务器是否已启动
                if "Local:" in output or "ready in" in output.lower():
                    print("\n✅ Vue开发服务器启动成功!")
                    print("🌐 请在浏览器中访问: http://localhost:3000")
                    
                    # 等待一秒后打开浏览器
                    time.sleep(2)
                    try:
                        webbrowser.open('http://localhost:3000')
                        print("📱 已自动打开浏览器")
                    except:
                        print("💡 请手动在浏览器中打开 http://localhost:3000")
                    
                    return True
                    
            except:
                pass
            
            time.sleep(1)
        
        print("⏰ 服务器启动超时")
        process.terminate()
        return False
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在停止服务器...")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

def print_vue_features():
    """打印Vue.js前端特性"""
    print("\n🎉 Vue.js 现代化前端特性:")
    print("=" * 40)
    print("✨ Vue 3 + Composition API")
    print("🎨 Element Plus UI组件库")
    print("📊 Chart.js 图表库集成")
    print("🗺️ Vue-Leaflet 地图组件")
    print("🔄 Pinia 状态管理")
    print("🛣️ Vue Router 路由系统")
    print("⚡ Vite 构建工具")
    print("🎯 TypeScript 支持")
    print("📱 响应式设计")
    print("🔧 热重载开发")
    print("=" * 40)
    
    print("\n📋 页面结构:")
    print("📊 仪表板 - 数据概览和统计")
    print("📡 实时监测 - 水质指标实时监控")
    print("🔮 智能预测 - AI预测服务")
    print("📈 数据分析 - 图表和趋势分析")
    print("✅ 数据质量 - 数据质量监控")
    print("🗺️ 空间分析 - 地图和空间查询")
    
    print("\n🎨 设计特点:")
    print("🌈 现代化渐变设计")
    print("🪟 玻璃态效果")
    print("⚡ 流畅动画过渡")
    print("📱 完全响应式")
    print("🎯 交互式组件")

def main():
    print("🔧 Vue.js 前端测试工具")
    print("=" * 50)
    
    # 检查环境
    if not check_node_npm():
        return False
    
    # 设置Vue项目
    if not setup_vue_project():
        return False
    
    # 检查后端
    backend_running = check_backend()
    if not backend_running:
        print("\n💡 提示: 前端可以独立启动，但需要后端才能获取数据")
        print("💡 启动后端: 在另一个终端运行 'python run.py'")
        print("⏳ 3秒后自动继续...")
        time.sleep(3)
    
    # 显示特性
    print_vue_features()
    
    # 启动Vue开发服务器
    return start_vue_dev_server()

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Vue.js前端启动成功!")
            print("🔄 开发服务器正在运行中...")
            print("📝 修改代码会自动热重载")
            print("🛑 按 Ctrl+C 停止服务器")
            try:
                # 保持脚本运行
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 感谢使用Vue.js前端!")
        else:
            print("\n❌ Vue.js前端启动失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        sys.exit(1)
