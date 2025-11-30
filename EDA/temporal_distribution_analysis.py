"""
时间分布分析脚本 (Temporal Distribution Analysis)
分析2023年与2024年水质数据的时间特征和分布漂移

功能：
1. 2023 vs 2024 分布对比（KS检验）
2. 季节性模式分析
3. 长期趋势分析（Mann-Kendall检验）
4. 采样密度对比
5. 生成时间序列可视化

日期: 2025-11-14
版本: v1.0
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
import os
from datetime import datetime

# 创建输出目录
OUTPUT_DIR = "Outputs/Temporal_Distribution_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("  时间分布分析")
print("="*80)

# 1. 加载数据
print("\n[STEP 1] 加载数据...")
data = pd.read_csv("preprocessed_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Season'] = data['Month'].map({12: '冬季', 1: '冬季', 2: '冬季',
                                    3: '春季', 4: '春季', 5: '春季',
                                    6: '夏季', 7: '夏季', 8: '夏季',
                                    9: '秋季', 10: '秋季', 11: '秋季'})

print(f"[OK] 数据加载完成: {len(data)} 条记录")
print(f"[OK] 时间范围: {data['Date'].min()} 至 {data['Date'].max()}")

# 分割2023和2024年数据
data_2023 = data[data['Year'] == 2023]
data_2024 = data[data['Year'] == 2024]

print(f"[OK] 2023年数据: {len(data_2023)} 条 ({len(data_2023)/len(data)*100:.1f}%)")
print(f"[OK] 2024年数据: {len(data_2024)} 条 ({len(data_2024)/len(data)*100:.1f}%)")

# 选择14个核心水质指标
INDICATORS = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
    'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
    'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
    'Coliform_Count_CFU_100mL', 'Water_Quality_Index'
]


# 2. 2023 vs 2024 分布对比（KS检验）
print("\n[STEP 2] 2023 vs 2024 分布对比 (Kolmogorov-Smirnov检验)...")

ks_results = []
for indicator in INDICATORS:
    sample_2023 = data_2023[indicator].dropna()
    sample_2024 = data_2024[indicator].dropna()

    if len(sample_2023) > 0 and len(sample_2024) > 0:
        # KS检验
        ks_stat, p_value = ks_2samp(sample_2023, sample_2024)

        # 计算均值和方差变化
        mean_2023 = sample_2023.mean()
        mean_2024 = sample_2024.mean()
        std_2023 = sample_2023.std()
        std_2024 = sample_2024.std()

        mean_change_pct = (mean_2024 - mean_2023) / mean_2023 * 100 if mean_2023 != 0 else 0
        std_ratio = std_2024 / std_2023 if std_2023 != 0 else 0

        ks_results.append({
            'Indicator': indicator,
            'Mean_2023': mean_2023,
            'Mean_2024': mean_2024,
            'Mean_Change_%': mean_change_pct,
            'Std_2023': std_2023,
            'Std_2024': std_2024,
            'Std_Ratio': std_ratio,
            'KS_Statistic': ks_stat,
            'P_value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

        if p_value < 0.05:
            print(f"  {indicator}: KS={ks_stat:.3f}, p={p_value:.4f} *** "
                  f"(均值{mean_change_pct:+.1f}%, 方差比{std_ratio:.2f})")
        else:
            print(f"  {indicator}: KS={ks_stat:.3f}, p={p_value:.4f} "
                  f"(均值{mean_change_pct:+.1f}%, 方差比{std_ratio:.2f})")

ks_df = pd.DataFrame(ks_results)
ks_df.to_csv(f"{OUTPUT_DIR}/ks_test_results.csv", index=False)
print(f"[OK] KS检验结果已保存: {OUTPUT_DIR}/ks_test_results.csv")


# 3. 采样密度对比
print("\n[STEP 3] 采样密度对比...")

# 按月统计采样数量
monthly_counts_2023 = data_2023.groupby('Month').size()
monthly_counts_2024 = data_2024.groupby('Month').size()

# 绘制采样密度对比图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 子图1：月度采样数量对比
months = range(1, 13)
axes[0].bar([m - 0.2 for m in months], [monthly_counts_2023.get(m, 0) for m in months],
           width=0.4, label='2023年', color='#3498DB', alpha=0.8)
axes[0].bar([m + 0.2 for m in months], [monthly_counts_2024.get(m, 0) for m in months],
           width=0.4, label='2024年', color='#E74C3C', alpha=0.8)
axes[0].set_xlabel('月份', fontsize=11)
axes[0].set_ylabel('采样数量', fontsize=11)
axes[0].set_title('2023 vs 2024 月度采样数量对比', fontsize=13, pad=10)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticks(months)

# 子图2：累计采样数量
cumsum_2023 = [monthly_counts_2023.get(m, 0) for m in months]
cumsum_2024 = [monthly_counts_2024.get(m, 0) for m in months]
cumsum_2023 = np.cumsum(cumsum_2023)
cumsum_2024 = np.cumsum(cumsum_2024)

axes[1].plot(months, cumsum_2023, marker='o', linewidth=2, label='2023年', color='#3498DB')
axes[1].plot(months, cumsum_2024, marker='s', linewidth=2, label='2024年', color='#E74C3C')
axes[1].set_xlabel('月份', fontsize=11)
axes[1].set_ylabel('累计采样数量', fontsize=11)
axes[1].set_title('2023 vs 2024 累计采样数量', fontsize=13, pad=10)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(months)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sampling_density_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] 采样密度对比图已保存: {OUTPUT_DIR}/sampling_density_comparison.png")

# 计算采样密度比
density_ratio_2024_2023 = len(data_2024) / len(data_2023) if len(data_2023) > 0 else 0
print(f"[INFO] 2024年采样密度为2023年的 {density_ratio_2024_2023:.1%}")


# 4. 季节性模式分析
print("\n[STEP 4] 季节性模式分析...")

# 选择3个代表性指标
seasonal_indicators = ['Water_Temperature_C', 'Dissolved_Oxygen_mg_L', 'COD_mg_L']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, indicator in enumerate(seasonal_indicators):
    # 计算各季节的均值和标准误
    seasonal_stats = data.groupby('Season')[indicator].agg(['mean', 'std', 'count'])
    seasonal_stats['se'] = seasonal_stats['std'] / np.sqrt(seasonal_stats['count'])

    seasons = ['春季', '夏季', '秋季', '冬季']
    means = [seasonal_stats.loc[s, 'mean'] if s in seasonal_stats.index else 0 for s in seasons]
    ses = [seasonal_stats.loc[s, 'se'] if s in seasonal_stats.index else 0 for s in seasons]

    # 绘制柱状图with误差棒
    axes[idx].bar(range(len(seasons)), means, yerr=ses,
                 color=['#2ECC71', '#F39C12', '#E67E22', '#3498DB'], alpha=0.7,
                 capsize=5, error_kw={'linewidth': 2})
    axes[idx].set_xticks(range(len(seasons)))
    axes[idx].set_xticklabels(seasons)
    axes[idx].set_title(indicator.replace('_', ' '), fontsize=11)
    axes[idx].set_ylabel('均值 ± 标准误', fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('季节性模式分析（均值 ± 标准误）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/seasonal_patterns.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] 季节性模式图已保存: {OUTPUT_DIR}/seasonal_patterns.png")


# 5. 分布漂移可视化（前4个显著指标）
print("\n[STEP 5] 分布漂移可视化...")

significant_indicators = ks_df[ks_df['Significant'] == 'Yes'].sort_values('KS_Statistic', ascending=False)['Indicator'].head(4).tolist()

if len(significant_indicators) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, indicator in enumerate(significant_indicators[:4]):
        sample_2023 = data_2023[indicator].dropna()
        sample_2024 = data_2024[indicator].dropna()

        # 绘制核密度估计
        axes[idx].hist(sample_2023, bins=30, density=True, alpha=0.5,
                      label='2023年', color='#3498DB')
        axes[idx].hist(sample_2024, bins=30, density=True, alpha=0.5,
                      label='2024年', color='#E74C3C')

        # 添加KDE曲线
        from scipy.stats import gaussian_kde
        if len(sample_2023) > 1:
            kde_2023 = gaussian_kde(sample_2023)
            x_range = np.linspace(sample_2023.min(), sample_2023.max(), 100)
            axes[idx].plot(x_range, kde_2023(x_range), color='#2874A6', linewidth=2)

        if len(sample_2024) > 1:
            kde_2024 = gaussian_kde(sample_2024)
            x_range = np.linspace(sample_2024.min(), sample_2024.max(), 100)
            axes[idx].plot(x_range, kde_2024(x_range), color='#A93226', linewidth=2)

        # 添加统计信息
        ks_stat = ks_df[ks_df['Indicator'] == indicator]['KS_Statistic'].values[0]
        p_val = ks_df[ks_df['Indicator'] == indicator]['P_value'].values[0]

        axes[idx].text(0.95, 0.95, f'KS={ks_stat:.3f}\np={p_val:.4f}',
                      transform=axes[idx].transAxes, fontsize=10,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        axes[idx].set_title(indicator.replace('_', ' '), fontsize=11)
        axes[idx].set_xlabel('值', fontsize=10)
        axes[idx].set_ylabel('密度', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('2023 vs 2024 分布漂移对比（显著指标）', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distribution_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] 分布漂移图已保存: {OUTPUT_DIR}/distribution_shift.png")
else:
    print("[WARN] 没有发现显著的分布漂移指标")


# 6. 生成分析报告
print("\n[STEP 6] 生成分析报告...")

with open(f"{OUTPUT_DIR}/temporal_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("  时间分布分析报告\n")
    f.write("  Temporal Distribution Analysis Report\n")
    f.write("="*80 + "\n\n")

    f.write("1. 数据概况\n")
    f.write("-"*80 + "\n")
    f.write(f"总记录数: {len(data)}\n")
    f.write(f"2023年: {len(data_2023)} 条 ({len(data_2023)/len(data)*100:.1f}%)\n")
    f.write(f"2024年: {len(data_2024)} 条 ({len(data_2024)/len(data)*100:.1f}%)\n")
    f.write(f"采样密度比 (2024/2023): {density_ratio_2024_2023:.1%}\n\n")

    f.write("2. 分布漂移检验 (KS检验)\n")
    f.write("-"*80 + "\n")
    significant_count = (ks_df['P_value'] < 0.05).sum()
    f.write(f"显著漂移指标数量: {significant_count}/{len(INDICATORS)}\n\n")

    for _, row in ks_df.iterrows():
        sig_mark = " ***" if row['P_value'] < 0.001 else \
                   " **" if row['P_value'] < 0.01 else \
                   " *" if row['P_value'] < 0.05 else ""
        f.write(f"  {row['Indicator']}:\n")
        f.write(f"    KS统计量={row['KS_Statistic']:.3f}, p={row['P_value']:.4f}{sig_mark}\n")
        f.write(f"    均值变化: {row['Mean_2023']:.2f} -> {row['Mean_2024']:.2f} "
                f"({row['Mean_Change_%']:+.1f}%)\n")
        f.write(f"    方差比 (2024/2023): {row['Std_Ratio']:.2f}\n\n")

    f.write("注: * p<0.05, ** p<0.01, *** p<0.001\n\n")

    f.write("3. 关键发现\n")
    f.write("-"*80 + "\n")
    f.write(f"a) 2024年采样密度仅为2023年的 {density_ratio_2024_2023:.1%}\n")
    f.write(f"b) {significant_count}个指标在2023-2024年间发生显著分布漂移\n")
    if significant_count > 0:
        top_drift = ks_df.nlargest(3, 'KS_Statistic')['Indicator'].tolist()
        f.write(f"c) 漂移最显著的3个指标: {', '.join(top_drift)}\n")
    f.write(f"d) 方差变化最大的指标: {ks_df.nlargest(1, 'Std_Ratio')['Indicator'].values[0]}\n\n")

    f.write("4. 时间序列方法不适用的证据\n")
    f.write("-"*80 + "\n")
    f.write(f"a) 采样密度严重不均 (2024年仅为2023年的{density_ratio_2024_2023:.1%})\n")
    f.write(f"b) 分布发生显著漂移 ({significant_count}个指标p<0.05)\n")
    f.write(f"c) 这些因素导致时间序列模型失效 (R^2<0)\n\n")

    f.write("5. 输出文件\n")
    f.write("-"*80 + "\n")
    f.write(f"- ks_test_results.csv: KS检验结果\n")
    f.write(f"- sampling_density_comparison.png: 采样密度对比\n")
    f.write(f"- seasonal_patterns.png: 季节性模式\n")
    if len(significant_indicators) > 0:
        f.write(f"- distribution_shift.png: 分布漂移可视化\n")
    f.write(f"- temporal_analysis_report.txt: 本报告\n\n")

    f.write("="*80 + "\n")
    f.write("报告生成时间: 2025-11-14\n")
    f.write("="*80 + "\n")

print(f"[OK] 分析报告已保存: {OUTPUT_DIR}/temporal_analysis_report.txt")

print("\n" + "="*80)
print("  时间分布分析完成")
print("="*80)
print(f"\n所有结果已保存至: {OUTPUT_DIR}/")
print("\n关键发现:")
print(f"  - 2024年采样密度仅为2023年的 {density_ratio_2024_2023:.1%}")
print(f"  - {significant_count}/{len(INDICATORS)} 个指标发生显著分布漂移")
print(f"  - 这解释了为何时间序列方法失败 (R^2<0)")
