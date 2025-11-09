#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM + 随机森林混合模型 - 批量运行脚本
Batch runner for LSTM + RF hybrid models

用法：
  python run_hybrid_models.py          # 运行所有模型
  python run_hybrid_models.py province # 只运行省份级别模型
  python run_hybrid_models.py station  # 只运行站点级别模型

作者：Claude Code
日期：2025-11-09
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

class HybridModelRunner:
    """混合模型运行器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.start_time = datetime.now()
        self.results = {}

    def print_header(self, title):
        """打印标题"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70 + "\n")

    def print_section(self, title):
        """打印分隔符"""
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}\n")

    def run_province_model(self):
        """运行省份级别模型"""
        self.print_section("🌍 运行省份级别 LSTM+RF 混合模型")

        model_path = self.project_root / "timeseries_models_province_revised" / "lstm_rf_hybrid_province.py"

        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False

        try:
            print(f"📂 模型位置: {model_path}")
            print(f"🚀 启动模型...\n")

            start = time.time()
            # 从项目根目录运行模型
            result = subprocess.run(
                [sys.executable, str(model_path)],
                cwd=str(self.project_root.parent),  # 从根目录运行
                capture_output=False
            )

            elapsed = time.time() - start
            self.results['province'] = {
                'status': 'success' if result.returncode == 0 else 'failed',
                'time': elapsed,
                'returncode': result.returncode
            }

            if result.returncode == 0:
                print(f"\n✅ 省份级别模型运行完成！耗时: {elapsed:.1f}秒")
                return True
            else:
                print(f"\n❌ 省份级别模型运行失败 (返回码: {result.returncode})")
                return False

        except Exception as e:
            print(f"❌ 运行出错: {e}")
            self.results['province'] = {'status': 'error', 'error': str(e)}
            return False

    def run_station_model(self):
        """运行站点级别模型"""
        self.print_section("🏢 运行站点级别 LSTM+RF 混合模型")

        model_path = self.project_root / "timeseries_models_station_revised" / "lstm_rf_hybrid_station.py"

        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False

        try:
            print(f"📂 模型位置: {model_path}")
            print(f"🚀 启动模型...\n")

            start = time.time()
            # 从项目根目录运行模型
            result = subprocess.run(
                [sys.executable, str(model_path)],
                cwd=str(self.project_root.parent),  # 从根目录运行
                capture_output=False
            )

            elapsed = time.time() - start
            self.results['station'] = {
                'status': 'success' if result.returncode == 0 else 'failed',
                'time': elapsed,
                'returncode': result.returncode
            }

            if result.returncode == 0:
                print(f"\n✅ 站点级别模型运行完成！耗时: {elapsed:.1f}秒")
                return True
            else:
                print(f"\n❌ 站点级别模型运行失败 (返回码: {result.returncode})")
                return False

        except Exception as e:
            print(f"❌ 运行出错: {e}")
            self.results['station'] = {'status': 'error', 'error': str(e)}
            return False

    def print_summary(self):
        """打印总结"""
        self.print_header("📋 运行总结")

        total_time = (datetime.now() - self.start_time).total_seconds()

        print("模型运行结果：\n")
        for model_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            if model_name == 'province':
                display_name = "省份级别模型"
            else:
                display_name = "站点级别模型"

            time_str = f"{result['time']:.1f}秒" if 'time' in result else "N/A"
            print(f"  {status_icon} {display_name}: {result['status']}")
            print(f"     └─ 耗时: {time_str}")

        print(f"\n📊 总耗时: {total_time:.1f}秒")

        # 输出位置
        outputs_dir = self.project_root / "Outputs"
        if outputs_dir.exists():
            print(f"\n📁 输出目录:")
            print(f"   - {outputs_dir / 'LSTM_RF_Hybrid_Province'}")
            print(f"   - {outputs_dir / 'LSTM_RF_Hybrid_Station'}")

        print("\n" + "="*70)

    def check_data(self):
        """检查数据文件是否存在"""
        self.print_section("📥 检查数据文件")

        data_dir = self.project_root / "data"
        required_files = [
            '中国水污染监测数据集.csv',
            'preprocessed_data_feature_eng.csv'
        ]

        all_exist = True
        for file in required_files:
            file_path = data_dir / file
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file} ({file_size:.1f} MB)")
            else:
                print(f"  ❌ {file} (不存在)")
                all_exist = False

        if not all_exist:
            print("\n⚠️  部分数据文件缺失！")
            print("   请确保以下文件在 models/data/ 目录中:")
            for file in required_files:
                print(f"   - {file}")
            return False

        print("\n✅ 数据检查完成！")
        return True

    def check_dependencies(self):
        """检查依赖库"""
        self.print_section("📦 检查依赖库")

        dependencies = {
            'torch': 'PyTorch',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'sklearn': 'Scikit-learn',
            'matplotlib': 'Matplotlib'
        }

        all_installed = True
        for module, name in dependencies.items():
            try:
                __import__(module)
                print(f"  ✅ {name}")
            except ImportError:
                print(f"  ❌ {name} (未安装)")
                all_installed = False

        if not all_installed:
            print("\n⚠️  部分依赖库未安装！")
            print("   请运行:")
            print("   pip install torch pandas numpy scikit-learn matplotlib")
            return False

        print("\n✅ 依赖检查完成！")
        return True

    def run(self, mode='all'):
        """主运行方法"""
        self.print_header("LSTM + 随机森林混合模型 - 批量运行")

        # 检查依赖
        if not self.check_dependencies():
            print("\n❌ 依赖不完整，请先安装所需库")
            return False

        # 检查数据
        if not self.check_data():
            print("\n❌ 数据文件缺失，请确保数据在正确位置")
            return False

        # 运行模型
        print("\n" + "="*70)
        print("开始运行模型...\n")

        success = True
        if mode in ['all', 'province']:
            if not self.run_province_model():
                success = False

        if mode in ['all', 'station']:
            if not self.run_station_model():
                success = False

        # 打印总结
        self.print_summary()

        return success

def main():
    """主程序"""
    runner = HybridModelRunner()

    # 解析命令行参数
    mode = 'all'
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['province', 'station']:
            mode = arg
        elif arg in ['-h', '--help', 'help']:
            print(__doc__)
            sys.exit(0)
        else:
            print(f"❌ 未知参数: {arg}")
            print(__doc__)
            sys.exit(1)

    # 运行
    success = runner.run(mode)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
