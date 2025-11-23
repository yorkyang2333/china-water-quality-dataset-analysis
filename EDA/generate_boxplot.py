#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成主要水质指标箱线图
Generate Boxplot for Main Water Quality Indicators
"""

import pandas as pd
import matplotlib.pyplot as plt
import zhplot  # 自动配置中文字体
import os

# 配置参数
INPUT_FILE = 'preprocessed_data.csv'
OUTPUT_DIR = 'Outputs/Boxplot_Analysis'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'indicator_boxplot.png')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(" 主要水质指标箱线图生成")
print("=" * 80)

# 加载数据
print(f"\n[STEP 1] 加载数据: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')
print(f"[OK] 数据加载完成: {len(df)} 条记录")

# 选择主要指标（与论文表5对应）
indicators = [
    ('Water_Temperature_C', '水温 (°C)'),
    ('pH', 'pH'),
    ('Dissolved_Oxygen_mg_L', '溶解氧 (mg/L)'),
    ('Conductivity_uS_cm', '电导率 (μS/cm)'),
    ('Turbidity_NTU', '浊度 (NTU)'),
    ('Ammonia_N_mg_L', '氨氮 (mg/L)'),
    ('Total_Phosphorus_mg_L', '总磷 (mg/L)'),
    ('Total_Nitrogen_mg_L', '总氮 (mg/L)'),
    ('COD_mg_L', 'COD (mg/L)'),
    ('BOD_mg_L', 'BOD (mg/L)')
]

print(f"\n[STEP 2] 准备数据...")
# 提取数据并重命名列
data_for_plot = []
labels = []
for col_name, label in indicators:
    if col_name in df.columns:
        # 归一化数据（使用Z-score标准化，便于在同一图表中展示）
        values = df[col_name].dropna()
        data_for_plot.append(values)
        labels.append(label)
        print(f"  - {label}: {len(values)} 个有效值")

print(f"[OK] 数据准备完成，共 {len(data_for_plot)} 个指标")

# 绘制箱线图
print(f"\n[STEP 3] 绘制箱线图...")
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制箱线图
bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='green', linewidth=2, linestyle='--'),
                whiskerprops=dict(color='gray', linewidth=1.5),
                capprops=dict(color='gray', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='orange', markersize=4,
                               linestyle='none', markeredgecolor='darkorange', alpha=0.5))

# 设置标题和标签
ax.set_title('主要水质指标分布箱线图', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('水质指标', fontsize=13, fontweight='bold')
ax.set_ylabel('指标数值', fontsize=13, fontweight='bold')

# 旋转x轴标签以避免重叠
plt.xticks(rotation=45, ha='right')

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='中位数'),
    Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='平均值'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
           markersize=6, label='异常值')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# 保存图片
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] 箱线图已保存: {OUTPUT_FILE}")

# 生成描述性统计报告
report_file = os.path.join(OUTPUT_DIR, 'boxplot_statistics.txt')
print(f"\n[STEP 4] 生成统计报告...")

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("主要水质指标描述性统计\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"样本量: {len(df)}\n\n")

    for col_name, label in indicators:
        if col_name in df.columns:
            values = df[col_name].dropna()
            f.write(f"\n{label}:\n")
            f.write(f"  样本数: {len(values)}\n")
            f.write(f"  均值: {values.mean():.2f}\n")
            f.write(f"  标准差: {values.std():.2f}\n")
            f.write(f"  最小值: {values.min():.2f}\n")
            f.write(f"  25%分位: {values.quantile(0.25):.2f}\n")
            f.write(f"  中位数: {values.median():.2f}\n")
            f.write(f"  75%分位: {values.quantile(0.75):.2f}\n")
            f.write(f"  最大值: {values.max():.2f}\n")
            f.write(f"  偏度: {values.skew():.2f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("报告结束\n")
    f.write("=" * 80 + "\n")

print(f"[OK] 统计报告已保存: {report_file}")

print("\n" + "=" * 80)
print(" 箱线图生成完成")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - 箱线图: {OUTPUT_FILE}")
print(f"  - 统计报告: {report_file}")
print()
