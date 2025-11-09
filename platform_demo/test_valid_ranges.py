#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证 valid_ranges 功能是否正确实现
"""

import sys
import os

# 添加后端路径
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from data.valid_ranges import check_value_in_range, check_all_values, VALID_RANGES

def test_single_range_check():
    """测试单个指标范围检查"""
    print("=" * 60)
    print("🧪 测试 1: 单个指标范围检查")
    print("=" * 60)

    # 测试用例1: 正常值
    result = check_value_in_range('pH', 7.0)
    assert result['in_range'] == True, "pH=7.0 应该在范围内"
    print("✓ pH=7.0 (正常值): 通过")

    # 测试用例2: 超出范围（过低）
    result = check_value_in_range('pH', 3.0)
    assert result['in_range'] == False, "pH=3.0 应该超出范围"
    print("✓ pH=3.0 (过低): 正确识别为超出范围")

    # 测试用例3: 超出范围（过高）
    result = check_value_in_range('COD_mg_L', 100.0)
    assert result['in_range'] == False, "COD=100 应该超出范围"
    print("✓ COD_mg_L=100 (过高): 正确识别为超出范围")
    print()


def test_all_values_check():
    """测试批量值检查"""
    print("=" * 60)
    print("🧪 测试 2: 批量值检查")
    print("=" * 60)

    # 正常数据
    normal_data = {
        'pH': 7.2,
        'COD_mg_L': 20.0,
        'Water_Temperature_C': 20.0
    }
    warnings = check_all_values(normal_data)
    assert len(warnings) == 0, "正常数据不应该有警告"
    print("✓ 正常数据: 无警告")

    # 包含异常值的数据
    abnormal_data = {
        'pH': 3.0,  # 超低
        'COD_mg_L': 100.0,  # 超高
        'Water_Temperature_C': 20.0  # 正常
    }
    warnings = check_all_values(abnormal_data)
    assert len(warnings) == 2, f"应该有2条警告，实际: {len(warnings)}"
    print(f"✓ 异常数据: 识别到 {len(warnings)} 条警告")
    for warning in warnings:
        print(f"  - {warning['indicator']}: {warning['message'].split(chr(10))[0]}")
    print()


def test_valid_ranges_definition():
    """测试范围定义的完整性"""
    print("=" * 60)
    print("🧪 测试 3: 范围定义完整性")
    print("=" * 60)

    required_indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    print(f"✓ 定义的指标数: {len(VALID_RANGES)}")
    print(f"✓ 需要的指标数: {len(required_indicators)}")

    for indicator in required_indicators:
        assert indicator in VALID_RANGES, f"缺少指标: {indicator}"

    print(f"✓ 所有 {len(required_indicators)} 个指标都已定义")

    # 打印几个范围作为验证
    print("\n✓ 范围示例:")
    print(f"  - pH: {VALID_RANGES['pH']['min']} ~ {VALID_RANGES['pH']['max']}")
    print(f"  - COD_mg_L: {VALID_RANGES['COD_mg_L']['min']} ~ {VALID_RANGES['COD_mg_L']['max']}")
    print(f"  - Water_Temperature_C: {VALID_RANGES['Water_Temperature_C']['min']} ~ {VALID_RANGES['Water_Temperature_C']['max']}")
    print()


def test_warning_messages():
    """测试警告信息格式"""
    print("=" * 60)
    print("🧪 测试 4: 警告信息格式")
    print("=" * 60)

    result = check_value_in_range('COD_mg_L', 100.0)
    assert result['in_range'] == False
    assert result['warning'] is not None
    assert '⚠️' in result['warning']
    assert 'COD_mg_L' in result['warning']
    assert '100' in result['warning']

    print("✓ 警告信息包含所有必要信息:")
    print("  ✓ 包含警告符号 (⚠️)")
    print("  ✓ 包含指标名称")
    print("  ✓ 包含实际值")
    print("\n✓ 警告信息示例:")
    for line in result['warning'].split('\n'):
        print(f"  {line}")
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  水质指标范围检查 - 功能验证")
    print("=" * 60 + "\n")

    try:
        test_single_range_check()
        test_all_values_check()
        test_valid_ranges_definition()
        test_warning_messages()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n✨ 范围检查功能工作正常:")
        print("  ✓ 单个值范围检查")
        print("  ✓ 批量值检查")
        print("  ✓ 范围定义完整")
        print("  ✓ 警告信息格式正确")
        print()
        return 0

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
