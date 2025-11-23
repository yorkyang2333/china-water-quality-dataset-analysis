#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成ACF自相关函数图
Generate ACF (Autocorrelation Function) Plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zhplot  # 自动配置中文字体
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import os

# 配置参数
INPUT_FILE = 'preprocessed_data.csv'
OUTPUT_DIR = 'Outputs/ACF_Analysis'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'acf_plot.png')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(" ACF自相关函数分析")
print("=" * 80)

# 加载数据
print(f"\n[STEP 1] 加载数据: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
print(f"[OK] 数据加载完成: {len(df)} 条记录")

# 选择一个关键指标进行ACF分析（使用COD作为代表）
indicator = 'COD_mg_L'
print(f"\n[STEP 2] 计算 {indicator} 的自相关函数...")

# 移除缺失值
series = df[indicator].dropna()
print(f"[INFO] 有效观测数: {len(series)}")

# 计算ACF（最多40个滞后阶数）
n_lags = min(40, len(series) // 2)
acf_values = acf(series, nlags=n_lags, fft=False)
print(f"[OK] ACF计算完成，滞后阶数: {n_lags}")
print(f"[INFO] ACF(1) = {acf_values[1]:.4f}")
print(f"[INFO] ACF(7) = {acf_values[7]:.4f}")
print(f"[INFO] ACF(30) = {acf_values[30]:.4f}")

# 绘制ACF图
print(f"\n[STEP 3] 绘制ACF图...")
fig, ax = plt.subplots(figsize=(12, 6))

# 使用statsmodels的plot_acf函数
plot_acf(series, lags=n_lags, ax=ax, alpha=0.05)

# 设置标题和标签
ax.set_title('自相关函数（ACF）图 - COD指标', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('滞后阶数（Lag）', fontsize=12)
ax.set_ylabel('自相关系数（ACF）', fontsize=12)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加关键数值标注
ax.text(0.02, 0.98, f'ACF(1) = {acf_values[1]:.4f}\n'
                      f'ACF(7) = {acf_values[7]:.4f}\n'
                      f'样本量 = {len(series)}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10)

# 保存图片
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] ACF图已保存: {OUTPUT_FILE}")

# 生成文本报告
report_file = os.path.join(OUTPUT_DIR, 'acf_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ACF自相关函数分析报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"分析指标: {indicator}\n")
    f.write(f"样本量: {len(series)}\n")
    f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("关键ACF值:\n")
    f.write(f"  ACF(1)  = {acf_values[1]:.4f}  (滞后1期)\n")
    f.write(f"  ACF(7)  = {acf_values[7]:.4f}  (滞后7期)\n")
    f.write(f"  ACF(30) = {acf_values[30]:.4f}  (滞后30期)\n\n")

    # 判断时间序列适用性
    f.write("时间序列适用性评估:\n")
    if abs(acf_values[1]) < 0.1:
        f.write("  ❌ ACF(1)接近零，表明相邻观测之间几乎没有线性相关性\n")
        f.write("  ❌ 数据不适合使用ARIMA、SARIMA等时间序列方法\n")
        f.write("  ✅ 建议使用横截面预测方法（如XGBoost、RandomForest）\n")
    elif abs(acf_values[1]) < 0.3:
        f.write("  ⚠️  ACF(1)较弱，时间依赖性不强\n")
        f.write("  ⚠️  时间序列方法可能效果有限\n")
    else:
        f.write("  ✅ ACF(1)显著，存在时间依赖性\n")
        f.write("  ✅ 可以尝试时间序列方法\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("报告结束\n")
    f.write("=" * 80 + "\n")

print(f"[OK] 分析报告已保存: {report_file}")

print("\n" + "=" * 80)
print(" ACF分析完成")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - ACF图: {OUTPUT_FILE}")
print(f"  - 分析报告: {report_file}")
print()
