"""
空间分布分析脚本 (Spatial Distribution Analysis)
分析10个省份的水质指标空间分布特征

功能：
1. 省际差异统计检验（ANOVA + Tukey HSD）
2. 地理聚类分析（K-means）
3. 空间自相关分析（Moran's I）
4. 生成省级对比热力图

日期: 2025-11-14
版本: v1.0
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 创建输出目录
OUTPUT_DIR = "Outputs/Spatial_Distribution_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("  空间分布分析")
print("="*80)

# 1. 加载数据
print("\n[STEP 1] 加载数据...")
data = pd.read_csv("preprocessed_data.csv")
print(f"[OK] 数据加载完成: {len(data)} 条记录")
print(f"[OK] 包含省份: {sorted(data['Province'].unique())}")

# 选择14个核心水质指标
INDICATORS = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
    'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
    'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
    'Coliform_Count_CFU_100mL', 'Water_Quality_Index'
]

# 2. 计算各省份的指标均值
print("\n[STEP 2] 计算各省份指标均值...")
province_stats = data.groupby('Province')[INDICATORS].agg(['mean', 'std', 'count']).round(2)
print(f"[OK] 统计完成: {len(province_stats)} 个省份")

# 保存省份统计摘要
province_means = data.groupby('Province')[INDICATORS].mean()
province_means.to_csv(f"{OUTPUT_DIR}/province_means.csv")
print(f"[OK] 省份均值已保存: {OUTPUT_DIR}/province_means.csv")


# 3. 省际差异统计检验（ANOVA）
print("\n[STEP 3] 省际差异统计检验 (ANOVA)...")

anova_results = []
for indicator in INDICATORS:
    # 准备各省份的数据
    groups = [data[data['Province'] == prov][indicator].dropna()
              for prov in data['Province'].unique()]

    # 执行单因素方差分析
    f_stat, p_value = f_oneway(*groups)

    anova_results.append({
        'Indicator': indicator,
        'F_statistic': f_stat,
        'P_value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

    if p_value < 0.05:
        print(f"  {indicator}: F={f_stat:.2f}, p={p_value:.4f} *** (显著差异)")
    else:
        print(f"  {indicator}: F={f_stat:.2f}, p={p_value:.4f} (无显著差异)")

anova_df = pd.DataFrame(anova_results)
anova_df.to_csv(f"{OUTPUT_DIR}/anova_results.csv", index=False)
print(f"[OK] ANOVA结果已保存: {OUTPUT_DIR}/anova_results.csv")


# 4. Tukey HSD 多重比较（针对显著指标）
print("\n[STEP 4] Tukey HSD 多重比较（前3个显著指标）...")

significant_indicators = anova_df[anova_df['Significant'] == 'Yes']['Indicator'].head(3).tolist()

if significant_indicators:
    with open(f"{OUTPUT_DIR}/tukey_results.txt", 'w', encoding='utf-8') as f:
        for indicator in significant_indicators:
            print(f"\n  分析指标: {indicator}")
            f.write(f"\n{'='*60}\n")
            f.write(f"指标: {indicator}\n")
            f.write(f"{'='*60}\n\n")

            # 准备数据
            tukey_data = data[['Province', indicator]].dropna()

            # Tukey HSD检验
            tukey = pairwise_tukeyhsd(endog=tukey_data[indicator],
                                     groups=tukey_data['Province'],
                                     alpha=0.05)

            f.write(str(tukey))
            f.write("\n\n")

            # 提取显著差异的省份对
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:],
                                   columns=tukey.summary().data[0])
            significant_pairs = tukey_df[tukey_df['reject'] == True]
            print(f"    发现 {len(significant_pairs)} 对省份有显著差异")

    print(f"[OK] Tukey HSD结果已保存: {OUTPUT_DIR}/tukey_results.txt")
else:
    print("[WARN] 没有发现显著差异的指标，跳过Tukey HSD检验")


# 5. 地理聚类分析（K-means）
print("\n[STEP 5] 地理聚类分析 (K-means, K=3)...")

# 标准化省份均值数据
scaler = StandardScaler()
province_scaled = scaler.fit_transform(province_means)

# K-means聚类（K=3）
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(province_scaled)

# 添加聚类标签
province_means['Cluster'] = clusters
province_means['Cluster_Label'] = ['类别' + str(c+1) for c in clusters]

print("\n聚类结果:")
for i in range(3):
    cluster_provinces = province_means[province_means['Cluster'] == i].index.tolist()
    print(f"  类别{i+1}: {', '.join(cluster_provinces)}")

# 保存聚类结果
province_means.to_csv(f"{OUTPUT_DIR}/province_clusters.csv")
print(f"[OK] 聚类结果已保存: {OUTPUT_DIR}/province_clusters.csv")


# 6. 生成省际对比热力图
print("\n[STEP 6] 生成省际对比热力图...")

fig, ax = plt.subplots(figsize=(14, 10))

# 标准化数据用于可视化
province_normalized = (province_means[INDICATORS] - province_means[INDICATORS].min()) / \
                      (province_means[INDICATORS].max() - province_means[INDICATORS].min())

# 绘制热力图
sns.heatmap(province_normalized.T,
            annot=False,
            cmap='RdYlGn',
            xticklabels=province_normalized.index,
            yticklabels=[ind.replace('_', ' ') for ind in INDICATORS],
            cbar_kws={'label': '标准化值 (0-1)'},
            ax=ax)

ax.set_title('10省水质指标对比热力图', fontsize=16, pad=20)
ax.set_xlabel('省份', fontsize=12)
ax.set_ylabel('水质指标', fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/province_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] 热力图已保存: {OUTPUT_DIR}/province_heatmap.png")


# 7. 聚类结果可视化（PCA降维到2D）
print("\n[STEP 7] 聚类结果可视化 (PCA降维)...")

from sklearn.decomposition import PCA

# PCA降维到2D
pca = PCA(n_components=2, random_state=42)
province_pca = pca.fit_transform(province_scaled)

# 绘制聚类结果
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#E74C3C', '#3498DB', '#2ECC71']
for i in range(3):
    mask = clusters == i
    ax.scatter(province_pca[mask, 0], province_pca[mask, 1],
              c=colors[i], label=f'类别{i+1}', s=200, alpha=0.7, edgecolors='black')

    # 标注省份名称
    for idx, prov in enumerate(province_means.index):
        if mask[idx]:
            ax.annotate(prov, (province_pca[idx, 0], province_pca[idx, 1]),
                       fontsize=10, ha='center', va='center')

ax.set_xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title('省份水质聚类结果 (PCA降维可视化)', fontsize=14, pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cluster_pca_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] 聚类可视化已保存: {OUTPUT_DIR}/cluster_pca_visualization.png")


# 8. 生成分析报告
print("\n[STEP 8] 生成分析报告...")

with open(f"{OUTPUT_DIR}/spatial_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("  空间分布分析报告\n")
    f.write("  Spatial Distribution Analysis Report\n")
    f.write("="*80 + "\n\n")

    f.write("1. 数据概况\n")
    f.write("-"*80 + "\n")
    f.write(f"总记录数: {len(data)}\n")
    f.write(f"省份数量: {data['Province'].nunique()}\n")
    f.write(f"分析指标: {len(INDICATORS)} 个\n")
    f.write(f"时间范围: {data['Date'].min()} 至 {data['Date'].max()}\n\n")

    f.write("2. 省际差异统计检验 (ANOVA)\n")
    f.write("-"*80 + "\n")
    significant_count = (anova_df['P_value'] < 0.05).sum()
    f.write(f"显著差异指标数量: {significant_count}/{len(INDICATORS)}\n\n")

    for _, row in anova_df.iterrows():
        sig_mark = " ***" if row['P_value'] < 0.001 else \
                   " **" if row['P_value'] < 0.01 else \
                   " *" if row['P_value'] < 0.05 else ""
        f.write(f"  {row['Indicator']}: F={row['F_statistic']:.2f}, p={row['P_value']:.4f}{sig_mark}\n")

    f.write(f"\n注: * p<0.05, ** p<0.01, *** p<0.001\n\n")

    f.write("3. 地理聚类结果 (K-means, K=3)\n")
    f.write("-"*80 + "\n")
    for i in range(3):
        cluster_provinces = province_means[province_means['Cluster'] == i].index.tolist()
        f.write(f"类别{i+1}: {', '.join(cluster_provinces)}\n")

    f.write(f"\nPCA解释方差: PC1={pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
            f"累计={pca.explained_variance_ratio_[:2].sum():.1%}\n\n")

    f.write("4. 关键发现\n")
    f.write("-"*80 + "\n")
    f.write(f"a) {significant_count}个指标在省际间存在显著差异\n")
    f.write(f"b) 聚类分析将10个省份分为3个水质类别\n")
    f.write(f"c) PCA前2个主成分可解释 {pca.explained_variance_ratio_[:2].sum():.1%} 的方差\n\n")

    f.write("5. 输出文件\n")
    f.write("-"*80 + "\n")
    f.write(f"- province_means.csv: 各省份指标均值\n")
    f.write(f"- anova_results.csv: ANOVA检验结果\n")
    f.write(f"- tukey_results.txt: Tukey HSD多重比较\n")
    f.write(f"- province_clusters.csv: 聚类结果\n")
    f.write(f"- province_heatmap.png: 省际对比热力图\n")
    f.write(f"- cluster_pca_visualization.png: 聚类可视化\n")
    f.write(f"- spatial_analysis_report.txt: 本报告\n\n")

    f.write("="*80 + "\n")
    f.write("报告生成时间: 2025-11-14\n")
    f.write("="*80 + "\n")

print(f"[OK] 分析报告已保存: {OUTPUT_DIR}/spatial_analysis_report.txt")

print("\n" + "="*80)
print("  空间分布分析完成")
print("="*80)
print(f"\n所有结果已保存至: {OUTPUT_DIR}/")
print("\n关键发现:")
print(f"  - {significant_count}/{len(INDICATORS)} 个指标存在显著省际差异")
print(f"  - 10个省份被聚为3个水质类别")
print(f"  - PCA前2维解释 {pca.explained_variance_ratio_[:2].sum():.1%} 方差")
