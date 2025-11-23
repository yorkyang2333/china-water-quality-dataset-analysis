"""
生成污染等级分布图
Generate Pollution Level Distribution Chart
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建输出目录
OUTPUT_DIR = 'Outputs/Pollution_Level_Distribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("生成污染等级分布图")
print("=" * 70)

# 加载数据
print("\n[1/2] 加载数据...")
df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
print(f"  - 数据形状: {df.shape}")
print(f"  - 污染等级分布:")
print(df['Pollution_Level'].value_counts())

# 生成图表
print("\n[2/2] 生成图表...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# 污染等级饼图
pollution_counts = df['Pollution_Level'].value_counts()
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
ax.pie(pollution_counts.values, labels=pollution_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 12})
ax.set_title('污染等级分布', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pollution_level_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ 图表已保存: {OUTPUT_DIR}/pollution_level_distribution.png")
print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
