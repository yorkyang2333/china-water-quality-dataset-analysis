#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成主要水质指标相关性热力图
用于论文第3.5节（图10）

输出：docs/paper/resources/chapter3_data_methods/3.1.5_correlation_heatmap.png
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

print('[INFO] 加载数据...')
# 加载预处理后的数据
df = pd.read_csv('preprocessed_data_feature_eng.csv')
print(f'[INFO] 数据集大小: {df.shape}')

# 16个核心水质指标
indicators = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Turbidity_NTU',
    'COD_mg_L', 'BOD_mg_L', 'Ammonia_N_mg_L',
    'Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L',
    'Nitrate_mg_L', 'Nitrite_mg_L',
    'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
    'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
]

print('[INFO] 计算Pearson相关系数矩阵...')
# 计算相关系数矩阵
corr_matrix = df[indicators].corr()

print('[INFO] 生成热力图...')
# 创建图形
fig, ax = plt.subplots(figsize=(16, 14))

# 绘制热力图
sns.heatmap(
    corr_matrix,
    annot=True,  # 显示相关系数数值
    fmt='.2f',   # 数值格式：保留2位小数
    cmap='RdBu_r',  # 配色方案：红色（正相关）到蓝色（负相关）
    center=0,    # 中心点为0
    vmin=-1,     # 最小值-1
    vmax=1,      # 最大值1
    square=True, # 方形单元格
    linewidths=0.5,  # 网格线宽度
    cbar_kws={'label': 'Pearson相关系数', 'shrink': 0.8},  # 颜色条标签
    annot_kws={'size': 10}  # 注释文本大小
)

# 中文标签映射
label_map = {
    'Water_Temperature_C': '水温',
    'pH': 'pH',
    'Dissolved_Oxygen_mg_L': '溶解氧',
    'Conductivity_uS_cm': '电导率',
    'Turbidity_NTU': '浊度',
    'COD_mg_L': 'COD',
    'BOD_mg_L': 'BOD',
    'Ammonia_N_mg_L': '氨氮',
    'Total_Nitrogen_mg_L': '总氮',
    'Total_Phosphorus_mg_L': '总磷',
    'Nitrate_mg_L': '硝酸盐',
    'Nitrite_mg_L': '亚硝酸盐',
    'Heavy_Metals_Pb_ug_L': '铅',
    'Heavy_Metals_Cd_ug_L': '镉',
    'Heavy_Metals_Hg_ug_L': '汞',
    'Coliform_Count_CFU_100mL': '大肠菌群'
}

# 设置中文标签
chinese_labels = [label_map[ind] for ind in indicators]
ax.set_xticklabels(chinese_labels, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(chinese_labels, rotation=0, fontsize=12)

# 标题
plt.title('主要水质指标相关性热力图（Pearson系数）', fontsize=18, pad=20, weight='bold')

# 紧凑布局
plt.tight_layout()

# 确保输出目录存在
output_dir = 'docs/paper/resources/chapter3_data_methods'
os.makedirs(output_dir, exist_ok=True)

# 保存图片
output_path = os.path.join(output_dir, '3.1.5_correlation_heatmap.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f'[DONE] 相关性热力图已保存至: {output_path}')
print(f'[INFO] 图片分辨率: 300 DPI')
print(f'[INFO] 图片尺寸: 16x14英寸')

# 输出关键相关性对
print('\n[INFO] 关键相关性对（|r| > 0.5）：')
for i in range(len(indicators)):
    for j in range(i+1, len(indicators)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            print(f'  {label_map[indicators[i]]} - {label_map[indicators[j]]}: r = {corr_val:.3f}')
