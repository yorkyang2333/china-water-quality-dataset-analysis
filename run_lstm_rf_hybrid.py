#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM + 随机森林混合模型 - 快速启动脚本
可直接在项目根目录运行

用法：
  python run_lstm_rf_hybrid.py              # 运行所有模型
  python run_lstm_rf_hybrid.py province     # 只运行省份级别
  python run_lstm_rf_hybrid.py station      # 只运行站点级别

作者：Claude Code
日期：2025-11-09
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """主程序"""
    mode = 'all'
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['province', 'station']:
            mode = arg
        elif arg in ['-h', '--help', 'help']:
            print(__doc__)
            return 0
        else:
            print(f"❌ 未知参数: {arg}\n")
            print(__doc__)
            return 1

    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    models_dir = project_root / 'models'

    print("\n" + "="*70)
    print("  🎯 LSTM + 随机森林混合模型 - 水质时间序列预测")
    print("="*70)

    # 检查数据文件
    print("\n📥 检查数据文件...")
    data_exists = False
    data_files = [
        'preprocessed_data_feature_eng.csv',
        '中国水污染监测数据集.csv',
        'preprocess/preprocessed_data_feature_eng.csv'
    ]

    for data_file in data_files:
        if (project_root / data_file).exists():
            print(f"  ✅ 找到: {data_file}")
            data_exists = True
            break

    if not data_exists:
        print("  ❌ 未找到数据文件！")
        print("\n  请确保以下文件之一在项目根目录:")
        for f in data_files:
            print(f"     - {f}")
        return 1

    # 运行省份级别模型
    if mode in ['all', 'province']:
        print("\n" + "-"*70)
        print("🌍 运行省份级别模型...\n")
        province_model = models_dir / "timeseries_models_province_revised" / "lstm_rf_hybrid_province.py"

        if not province_model.exists():
            print(f"  ❌ 找不到模型: {province_model}")
            return 1

        try:
            result = subprocess.run(
                [sys.executable, str(province_model)],
                cwd=str(project_root)
            )
            if result.returncode != 0:
                print(f"  ❌ 省份级别模型运行失败")
                return 1
            print("\n  ✅ 省份级别模型完成")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return 1

    # 运行站点级别模型
    if mode in ['all', 'station']:
        print("\n" + "-"*70)
        print("🏢 运行站点级别模型...\n")
        station_model = models_dir / "timeseries_models_station_revised" / "lstm_rf_hybrid_station.py"

        if not station_model.exists():
            print(f"  ❌ 找不到模型: {station_model}")
            return 1

        try:
            result = subprocess.run(
                [sys.executable, str(station_model)],
                cwd=str(project_root)
            )
            if result.returncode != 0:
                print(f"  ❌ 站点级别模型运行失败")
                return 1
            print("\n  ✅ 站点级别模型完成")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return 1

    print("\n" + "="*70)
    print("  ✅ 所有模型运行完成！")
    print("="*70)

    print("\n📊 输出位置:")
    print("   • Outputs/LSTM_RF_Hybrid_Province/ - 省份级别预测")
    print("   • Outputs/LSTM_RF_Hybrid_Station/  - 站点级别预测")

    print("\n📖 查看文档:")
    print("   • models/QUICK_START.md - 快速开始指南")
    print("   • models/HYBRID_MODEL_README.md - 完整手册")

    print("\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
