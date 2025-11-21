#!/usr/bin/env python3
"""
一键运行所有模型 - 快速启动脚本
Quick Start: Run All Models

功能:
1. 运行Transformer分类器
2. 运行Autoformer时序预测（可选）
3. 运行终极集成系统
4. 生成完整对比报告

使用方法:
    python run_all_ultimate_models.py
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("一键运行所有模型 - 终极集成系统")
print("=" * 80)


def run_script(script_path, description):
    """运行Python脚本"""
    print(f"\n{'='*80}")
    print(f"运行: {description}")
    print(f"脚本: {script_path}")
    print(f"{'='*80}\n")

    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print(f"\n[OK] {description} 完成")
            return True
        else:
            print(f"\n[ERROR] {description} 失败，退出码: {result.returncode}")
            return False

    except Exception as e:
        print(f"\n[ERROR] {description} 运行出错: {e}")
        return False


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='一键运行所有模型')
    parser.add_argument('--skip-timeseries', action='store_true',
                        help='跳过时序预测模型（因为效果不佳）')
    parser.add_argument('--only-ensemble', action='store_true',
                        help='只运行终极集成系统')
    args = parser.parse_args()

    success_count = 0
    total_count = 0

    # 1. Transformer分类器
    if not args.only_ensemble:
        total_count += 1
        script_path = project_root / 'models' / 'pollution_level_classification' / 'models' / 'transformer_classifier.py'
        if script_path.exists():
            if run_script(script_path, "Transformer水质分类器"):
                success_count += 1
        else:
            print(f"[WARN] 脚本不存在: {script_path}")

    # 2. Autoformer时序预测（可选）
    if not args.skip_timeseries and not args.only_ensemble:
        total_count += 1
        script_path = project_root / 'models' / 'timeseries_models_province_revised' / 'autoformer_province.py'
        if script_path.exists():
            print("\n[INFO] 注意: 根据项目文档，时序方法因数据采样不规则而效果不佳")
            print("[INFO] 此模型主要用于学术探索和对比实验")
            if run_script(script_path, "Autoformer时序预测"):
                success_count += 1
        else:
            print(f"[WARN] 脚本不存在: {script_path}")

    # 3. 终极集成系统
    total_count += 1
    script_path = project_root / 'models' / 'pollution_level_classification' / 'models' / 'ultimate_ensemble_system.py'
    if script_path.exists():
        if run_script(script_path, "终极集成预测系统"):
            success_count += 1
    else:
        print(f"[WARN] 脚本不存在: {script_path}")

    # 汇总
    print("\n" + "=" * 80)
    print("执行完成!")
    print("=" * 80)
    print(f"成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("\n[OK] 所有模型运行成功！")
        print("\n输出目录:")
        print("  - Transformer分类: Outputs/Transformer_Classification/")
        print("  - Autoformer时序:  Outputs/Autoformer_Province_Prediction/")
        print("  - 终极集成系统:    Outputs/Ultimate_Ensemble_System/")
    else:
        print(f"\n[WARN] 有 {total_count - success_count} 个模型运行失败")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
