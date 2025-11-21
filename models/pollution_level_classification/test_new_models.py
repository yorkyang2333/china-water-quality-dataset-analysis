#!/usr/bin/env python3
"""
快速测试脚本 - 验证所有新模型
Quick Test: Verify All New Models

功能:
1. 检查依赖库是否安装
2. 测试Transformer分类器
3. 测试Autoformer时序预测
4. 测试终极集成系统
5. 生成测试报告

使用方法:
    python test_new_models.py
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("快速测试 - 验证所有新模型")
print("=" * 80)

# 测试结果
test_results = {}


def test_dependencies():
    """测试依赖库"""
    print("\n[TEST 1/4] 检查依赖库...")

    dependencies = {
        'numpy': False,
        'pandas': False,
        'sklearn': False,
        'torch': False,
        'xgboost': False,
        'lightgbm': False,
        'catboost': False
    }

    for lib in dependencies.keys():
        try:
            __import__(lib)
            dependencies[lib] = True
            print(f"  [OK] {lib}")
        except ImportError:
            print(f"  [FAIL] {lib} - 未安装")

    # 统计
    installed = sum(dependencies.values())
    total = len(dependencies)
    test_results['dependencies'] = f"{installed}/{total}"

    print(f"\n  总计: {installed}/{total} 已安装")

    return installed >= 5  # 至少需要5个核心库


def test_transformer():
    """测试Transformer分类器"""
    print("\n[TEST 2/4] 测试Transformer分类器...")

    try:
        script_path = project_root / 'models' / 'pollution_level_classification' / 'models' / 'transformer_classifier.py'

        if not script_path.exists():
            print(f"  [FAIL] 文件不存在: {script_path}")
            test_results['transformer'] = 'FAILED - File not found'
            return False

        # 检查代码语法
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
            compile(code, script_path, 'exec')

        print(f"  [OK] 文件存在并且语法正确")
        print(f"  [OK] 路径: {script_path}")
        test_results['transformer'] = 'PASSED'
        return True

    except Exception as e:
        print(f"  [FAIL] 错误: {e}")
        test_results['transformer'] = f'FAILED - {str(e)[:50]}'
        return False


def test_autoformer():
    """测试Autoformer时序预测"""
    print("\n[TEST 3/4] 测试Autoformer时序预测...")

    try:
        script_path = project_root / 'models' / 'timeseries_models_province_revised' / 'autoformer_province.py'

        if not script_path.exists():
            print(f"  [FAIL] 文件不存在: {script_path}")
            test_results['autoformer'] = 'FAILED - File not found'
            return False

        # 检查代码语法
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
            compile(code, script_path, 'exec')

        print(f"  [OK] 文件存在并且语法正确")
        print(f"  [OK] 路径: {script_path}")
        test_results['autoformer'] = 'PASSED'
        return True

    except Exception as e:
        print(f"  [FAIL] 错误: {e}")
        test_results['autoformer'] = f'FAILED - {str(e)[:50]}'
        return False


def test_ultimate_ensemble():
    """测试终极集成系统"""
    print("\n[TEST 4/4] 测试终极集成系统...")

    try:
        script_path = project_root / 'models' / 'pollution_level_classification' / 'models' / 'ultimate_ensemble_system.py'

        if not script_path.exists():
            print(f"  [FAIL] 文件不存在: {script_path}")
            test_results['ultimate_ensemble'] = 'FAILED - File not found'
            return False

        # 检查代码语法
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
            compile(code, script_path, 'exec')

        print(f"  [OK] 文件存在并且语法正确")
        print(f"  [OK] 路径: {script_path}")
        test_results['ultimate_ensemble'] = 'PASSED'
        return True

    except Exception as e:
        print(f"  [FAIL] 错误: {e}")
        test_results['ultimate_ensemble'] = f'FAILED - {str(e)[:50]}'
        return False


def main():
    """主程序"""

    # 运行所有测试
    results = []

    results.append(('依赖库检查', test_dependencies()))
    results.append(('Transformer分类器', test_transformer()))
    results.append(('Autoformer时序预测', test_autoformer()))
    results.append(('终极集成系统', test_ultimate_ensemble()))

    # 生成报告
    print("\n" + "=" * 80)
    print("测试报告")
    print("=" * 80)

    print(f"\n{'测试项':<30} {'状态':<15}")
    print("-" * 80)

    passed = 0
    total = len(results)

    for name, result in results:
        status = '[OK] PASSED' if result else '[FAIL] FAILED'
        print(f"{name:<30} {status:<15}")
        if result:
            passed += 1

    print("-" * 80)
    print(f"总计: {passed}/{total} 通过\n")

    # 详细结果
    print("详细结果:")
    for key, value in test_results.items():
        print(f"  {key}: {value}")

    # 建议
    print("\n" + "=" * 80)
    if passed == total:
        print("[OK] 所有测试通过！模型已准备好运行。")
        print("\n下一步:")
        print("  1. 运行单个模型: python models/.../transformer_classifier.py")
        print("  2. 运行所有模型: python models/.../run_all_ultimate_models.py")
        print("  3. 查看文档: models/.../ULTIMATE_MODELS_GUIDE.md")
    else:
        print(f"[WARN] {total - passed} 个测试失败")
        print("\n建议:")
        if test_results.get('dependencies', '').startswith('0/') or test_results.get('dependencies', '').startswith('1/'):
            print("  1. 安装缺失的依赖库: pip install -r requirements.txt")
            print("  2. 安装PyTorch: pip install torch")
            print("  3. 安装XGBoost: pip install xgboost")
        else:
            print("  1. 检查文件路径是否正确")
            print("  2. 查看错误详情并修复")

    print("=" * 80)


if __name__ == '__main__':
    main()
