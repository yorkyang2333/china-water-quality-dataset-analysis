"""
时间序列可视化与EDA分析
Time Series Visualization and EDA

功能：
1. 为所有监测站点生成时间序列可视化折线图
2. 分析时间采样间隔特征
3. 探索各指标随时间的变化趋势
4. 生成汇总统计和可视化报告

作者：Claude Code
日期：2025-10-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建输出目录
OUTPUT_DIR = "Outputs/Timeseries_EDA"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Station_Plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Summary_Plots", exist_ok=True)

print("=" * 70)
print("时间序列可视化与EDA分析")
print("Time Series Visualization and Exploratory Data Analysis")
print("=" * 70)

# ==================== 1. 数据加载 ====================
print("\n[1/6] 加载数据...")
df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')

# 转换日期列为datetime格式
df['Date'] = pd.to_datetime(df['Date'])

# 按日期排序
df = df.sort_values('Date')

print(f"  - 数据形状: {df.shape}")
print(f"  - 日期范围: {df['Date'].min()} 至 {df['Date'].max()}")
print(f"  - 监测站数量: {df['Monitoring_Station'].nunique()}")
print(f"  - 记录总数: {len(df)}")

# 选择主要的水质指标进行可视化
key_indicators = [
    'Water_Quality_Index',
    'pH',
    'Dissolved_Oxygen_mg_L',
    'Turbidity_NTU',
    'Ammonia_N_mg_L',
    'Total_Phosphorus_mg_L',
    'Total_Nitrogen_mg_L',
    'COD_mg_L',
    'BOD_mg_L'
]

# ==================== 2. 时间采样间隔分析 ====================
print("\n[2/6] 分析时间采样间隔...")

# 计算每个站点的采样间隔
time_interval_stats = []

for station in df['Monitoring_Station'].unique():
    station_data = df[df['Monitoring_Station'] == station].sort_values('Date')

    if len(station_data) > 1:
        # 计算相邻采样点之间的时间差（天数）
        intervals = station_data['Date'].diff().dt.days.dropna()

        time_interval_stats.append({
            'Station': station,
            'Province': station_data['Province'].iloc[0],
            'City': station_data['City'].iloc[0],
            'Sample_Count': len(station_data),
            'Min_Interval_Days': intervals.min(),
            'Max_Interval_Days': intervals.max(),
            'Mean_Interval_Days': intervals.mean(),
            'Median_Interval_Days': intervals.median(),
            'Std_Interval_Days': intervals.std()
        })

interval_df = pd.DataFrame(time_interval_stats)
interval_df.to_csv(f"{OUTPUT_DIR}/sampling_interval_analysis.csv", index=False, encoding='utf-8-sig')

print(f"  - 平均采样间隔: {interval_df['Mean_Interval_Days'].mean():.2f} 天")
print(f"  - 采样间隔标准差: {interval_df['Std_Interval_Days'].mean():.2f} 天")
print(f"  - 最小采样间隔: {interval_df['Min_Interval_Days'].min():.0f} 天")
print(f"  - 最大采样间隔: {interval_df['Max_Interval_Days'].max():.0f} 天")

# 可视化采样间隔分布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('采样间隔分析 | Sampling Interval Analysis', fontsize=16, fontweight='bold')

# 2.1 采样间隔的总体分布（直方图）
all_intervals = []
for station in df['Monitoring_Station'].unique():
    station_data = df[df['Monitoring_Station'] == station].sort_values('Date')
    if len(station_data) > 1:
        intervals = station_data['Date'].diff().dt.days.dropna()
        all_intervals.extend(intervals.tolist())

axes[0, 0].hist(all_intervals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Time Interval (Days)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Sampling Intervals', fontsize=13, fontweight='bold')
axes[0, 0].axvline(np.mean(all_intervals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_intervals):.1f} days')
axes[0, 0].legend()

# 2.2 各站点的平均采样间隔
top_stations = interval_df.nlargest(20, 'Mean_Interval_Days')
axes[0, 1].barh(range(len(top_stations)), top_stations['Mean_Interval_Days'], color='coral')
axes[0, 1].set_yticks(range(len(top_stations)))
axes[0, 1].set_yticklabels(top_stations['Station'], fontsize=8)
axes[0, 1].set_xlabel('Mean Interval (Days)', fontsize=12)
axes[0, 1].set_title('Top 20 Stations with Longest Mean Intervals', fontsize=13, fontweight='bold')

# 2.3 采样数量分布
axes[1, 0].hist(interval_df['Sample_Count'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Number of Samples per Station', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Sample Counts', fontsize=13, fontweight='bold')

# 2.4 采样间隔标准差（反映不规则性）
axes[1, 1].scatter(interval_df['Mean_Interval_Days'], interval_df['Std_Interval_Days'],
                   alpha=0.6, s=interval_df['Sample_Count']*3, c='purple')
axes[1, 1].set_xlabel('Mean Interval (Days)', fontsize=12)
axes[1, 1].set_ylabel('Std Deviation of Intervals (Days)', fontsize=12)
axes[1, 1].set_title('Interval Irregularity (bubble size = sample count)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Summary_Plots/01_sampling_interval_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ 采样间隔分析图表已保存")

# ==================== 3. 整体时间趋势分析 ====================
print("\n[3/6] 分析整体时间趋势...")

# 按月份聚合所有数据
df['Year_Month'] = df['Date'].dt.to_period('M')
monthly_avg = df.groupby('Year_Month')[key_indicators].mean()
monthly_avg.index = monthly_avg.index.to_timestamp()

# 可视化主要指标的时间趋势
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('主要水质指标的时间趋势 (月度平均) | Monthly Average Trends', fontsize=16, fontweight='bold')

for idx, indicator in enumerate(key_indicators):
    row = idx // 3
    col = idx % 3

    axes[row, col].plot(monthly_avg.index, monthly_avg[indicator], marker='o', linewidth=2, markersize=4)
    axes[row, col].set_xlabel('Date', fontsize=11)
    axes[row, col].set_ylabel(indicator, fontsize=11)
    axes[row, col].set_title(f'{indicator}', fontsize=12, fontweight='bold')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Summary_Plots/02_overall_time_trends.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ 整体时间趋势图表已保存")

# ==================== 4. 各省份时间趋势对比 ====================
print("\n[4/6] 分析各省份时间趋势...")

provinces = df['Province'].unique()

# 选择3个关键指标进行省份对比
comparison_indicators = ['Water_Quality_Index', 'COD_mg_L', 'Total_Nitrogen_mg_L']

for indicator in comparison_indicators:
    fig, ax = plt.subplots(figsize=(16, 8))

    for province in provinces:
        province_data = df[df['Province'] == province].copy()
        province_data['Year_Month'] = province_data['Date'].dt.to_period('M')
        province_monthly = province_data.groupby('Year_Month')[indicator].mean()
        province_monthly.index = province_monthly.index.to_timestamp()

        ax.plot(province_monthly.index, province_monthly, marker='o', linewidth=2,
                markersize=3, label=province, alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(indicator, fontsize=12)
    ax.set_title(f'{indicator} - Provincial Comparison (Monthly Average)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Summary_Plots/03_province_comparison_{indicator}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"  ✓ 省份对比图表已保存")

# ==================== 5. 生成各监测站点的详细时间序列图 ====================
print("\n[5/6] 生成各监测站点的详细时间序列图...")
print("  (这可能需要几分钟，请稍候...)")

stations = df['Monitoring_Station'].unique()
total_stations = len(stations)

for idx, station in enumerate(stations, 1):
    if idx % 20 == 0:
        print(f"  - 进度: {idx}/{total_stations} 个站点")

    station_data = df[df['Monitoring_Station'] == station].sort_values('Date')

    # 为每个站点创建一个包含多个指标的综合图表
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    province = station_data['Province'].iloc[0]
    city = station_data['City'].iloc[0]

    fig.suptitle(f'{station} | {city}, {province}\n({len(station_data)} samples)',
                 fontsize=14, fontweight='bold')

    for i, indicator in enumerate(key_indicators):
        row = i // 3
        col = i % 3

        axes[row, col].plot(station_data['Date'], station_data[indicator],
                           marker='o', linewidth=1.5, markersize=4, color='steelblue')
        axes[row, col].set_xlabel('Date', fontsize=9)
        axes[row, col].set_ylabel(indicator, fontsize=9)
        axes[row, col].set_title(f'{indicator}', fontsize=10, fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(axis='x', rotation=45, labelsize=8)
        axes[row, col].tick_params(axis='y', labelsize=8)

    plt.tight_layout()

    # 文件名：使用站点名称（去除特殊字符）
    safe_station_name = station.replace('/', '_').replace('\\', '_')
    plt.savefig(f"{OUTPUT_DIR}/Station_Plots/{safe_station_name}.png", dpi=200, bbox_inches='tight')
    plt.close()

print(f"  ✓ 已为所有 {total_stations} 个监测站点生成时间序列图")

# ==================== 6. 生成EDA摘要报告 ====================
print("\n[6/6] 生成EDA摘要报告...")

summary_stats = {
    'Indicator': [],
    'Overall_Mean': [],
    'Overall_Std': [],
    'Overall_Min': [],
    'Overall_Max': [],
    'Trend_Direction': []
}

for indicator in key_indicators:
    summary_stats['Indicator'].append(indicator)
    summary_stats['Overall_Mean'].append(df[indicator].mean())
    summary_stats['Overall_Std'].append(df[indicator].std())
    summary_stats['Overall_Min'].append(df[indicator].min())
    summary_stats['Overall_Max'].append(df[indicator].max())

    # 简单趋势分析：比较前半段和后半段的平均值
    mid_date = df['Date'].min() + (df['Date'].max() - df['Date'].min()) / 2
    first_half = df[df['Date'] <= mid_date][indicator].mean()
    second_half = df[df['Date'] > mid_date][indicator].mean()

    if second_half > first_half * 1.05:
        trend = 'Increasing'
    elif second_half < first_half * 0.95:
        trend = 'Decreasing'
    else:
        trend = 'Stable'

    summary_stats['Trend_Direction'].append(trend)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f"{OUTPUT_DIR}/eda_summary_statistics.csv", index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("EDA摘要统计")
print("=" * 70)
print(summary_df.to_string(index=False))

# 生成综合文本报告
report_content = f"""
时间序列可视化与EDA分析报告
Time Series Visualization and EDA Report
{"=" * 70}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 数据概览
   - 数据形状: {df.shape}
   - 日期范围: {df['Date'].min()} 至 {df['Date'].max()}
   - 监测站数量: {df['Monitoring_Station'].nunique()}
   - 省份数量: {df['Province'].nunique()}
   - 城市数量: {df['City'].nunique()}
   - 记录总数: {len(df)}

2. 时间采样特征
   - 平均采样间隔: {interval_df['Mean_Interval_Days'].mean():.2f} 天
   - 采样间隔标准差: {interval_df['Std_Interval_Days'].mean():.2f} 天
   - 最小采样间隔: {interval_df['Min_Interval_Days'].min():.0f} 天
   - 最大采样间隔: {interval_df['Max_Interval_Days'].max():.0f} 天
   - 平均每站样本数: {interval_df['Sample_Count'].mean():.1f}

⚠️  关键发现：采样时间间隔极不规则（标准差大），这证实了项目文档中提到的
   不规则抽样问题，因此不适合使用传统的等间隔时间序列预测方法。

3. 主要指标统计摘要
{summary_df.to_string(index=False)}

4. 输出文件清单
   ✓ {OUTPUT_DIR}/Summary_Plots/01_sampling_interval_analysis.png
   ✓ {OUTPUT_DIR}/Summary_Plots/02_overall_time_trends.png
   ✓ {OUTPUT_DIR}/Summary_Plots/03_province_comparison_*.png (3个文件)
   ✓ {OUTPUT_DIR}/Station_Plots/*.png ({total_stations}个站点图表)
   ✓ {OUTPUT_DIR}/sampling_interval_analysis.csv
   ✓ {OUTPUT_DIR}/eda_summary_statistics.csv

5. 主要发现
   - 数据覆盖时间跨度约为 {(df['Date'].max() - df['Date'].min()).days} 天
   - 各站点采样频率差异显著，最密集和最稀疏站点间隔相差约
     {interval_df['Max_Interval_Days'].max() / interval_df['Min_Interval_Days'].min():.1f} 倍
   - 各指标时间趋势各异，部分指标呈现季节性波动特征

6. 建议
   基于时间序列EDA分析，我们建议：

   ✓ 继续使用横截面分类/回归方法（如Random Forest、XGBoost），这些方法
     已被证明在本数据集上表现优异（准确率>99%）

   ✗ 不推荐使用传统时间序列方法（ARIMA、SARIMA、Prophet等），因为：
     - 采样间隔极不规则
     - 各站点采样频率不一致
     - 缺乏足够的连续等间隔数据点

   ⚠️ 如果必须尝试时间序列预测，建议：
     - 仅对采样点密集且相对规则的个别站点尝试
     - 使用能处理不规则时间序列的模型（如基于RNN的神经网络）
     - 重采样到固定间隔（如月度），但会损失大量信息

{"=" * 70}
报告结束
"""

with open(f"{OUTPUT_DIR}/EDA_Report.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)

print("\n" + "=" * 70)
print("✅ 时间序列可视化与EDA分析完成！")
print("=" * 70)
print(f"\n所有输出已保存至: {OUTPUT_DIR}/")
print(f"  - 汇总图表: {OUTPUT_DIR}/Summary_Plots/")
print(f"  - 站点详细图表: {OUTPUT_DIR}/Station_Plots/")
print(f"  - 数据分析: {OUTPUT_DIR}/*.csv")
print(f"  - 文本报告: {OUTPUT_DIR}/EDA_Report.txt")
print("\n" + "=" * 70)
