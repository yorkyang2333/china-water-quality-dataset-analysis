#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM + 随机森林混合模型 - 直接运行脚本
可在项目根目录直接执行：python models/lstm_rf_hybrid_run.py

用法：
  python models/lstm_rf_hybrid_run.py               # 运行所有模型
  python models/lstm_rf_hybrid_run.py province      # 只运行省份级别
  python models/lstm_rf_hybrid_run.py station       # 只运行站点级别

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
        else:
            print(__doc__)
            return 1

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.absolute()
    models_dir = project_root / 'models'

    print("\n" + "="*70)
    print("  LSTM + 随机森林混合模型 - 水质时间序列预测")
    print("="*70)

    # 运行省份级别模型
    if mode in ['all', 'province']:
        print("\n🌍 运行省份级别模型...\n")
        province_model = models_dir / "timeseries_models_province_revised" / "lstm_rf_hybrid_province.py"

        if not province_model.exists():
            print(f"❌ 找不到模型: {province_model}")
            return 1

        try:
            result = subprocess.run(
                [sys.executable, str(province_model)],
                cwd=str(project_root)  # 从项目根目录运行
            )
            if result.returncode != 0:
                print(f"❌ 省份级别模型运行失败")
                return 1
        except Exception as e:
            print(f"❌ 错误: {e}")
            return 1

    # 运行站点级别模型
    if mode in ['all', 'station']:
        print("\n🏢 运行站点级别模型...\n")
        station_model = models_dir / "timeseries_models_station_revised" / "lstm_rf_hybrid_station.py"

        if not station_model.exists():
            print(f"❌ 找不到模型: {station_model}")
            return 1

        try:
            result = subprocess.run(
                [sys.executable, str(station_model)],
                cwd=str(project_root)  # 从项目根目录运行
            )
            if result.returncode != 0:
                print(f"❌ 站点级别模型运行失败")
                return 1
        except Exception as e:
            print(f"❌ 错误: {e}")
            return 1

    print("\n" + "="*70)
    print("✅ 所有模型运行完成！")
    print("="*70)
    print("\n📊 输出位置: Outputs/LSTM_RF_Hybrid_Province 和 Outputs/LSTM_RF_Hybrid_Station")
    print("\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
