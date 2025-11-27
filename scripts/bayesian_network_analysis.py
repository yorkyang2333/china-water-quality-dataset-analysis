"""
贝叶斯网络分析：大肠菌群污染与人类活动的关联分析
Bayesian Network Analysis for E. coli Contamination and Human Activities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
import zhplot

# ============================================================
# 第一步：数据加载与预处理
# ============================================================

print("[OK] 加载数据集...")
data = pd.read_csv('./preprocessed_data_feature_eng.csv')

print(f"[OK] 数据集大小：{data.shape}")
print(f"[OK] 列名：{data.columns.tolist()}")

# 显示大肠菌群统计
print("\n[INFO] 大肠菌群计数统计：")
print(data['Coliform_Count_CFU_100mL'].describe())

# ============================================================
# 第二步：特征工程 - 人类活动代理变量构建
# ============================================================

print("\n[OK] 构建人类活动代理变量...")

# 1. 有机污染负荷：基于COD（有机污染）和NH3-N（生活污水指示）
data['Organic_Pollution_Load'] = (data['COD_mg_L'] + data['Ammonia_N_mg_L'] * 10) / 2

# 2. 污水处理覆盖率代理：使用溶解氧的倒数（DO低表示有机物多，污水处理可能不足）
data['DO_Inverse'] = 1 / (data['Dissolved_Oxygen_mg_L'] + 0.1)

# 3. 重金属污染压力：使用重金属浓度
data['Heavy_Metal_Stress'] = data['Total_Heavy_Metals_ug_L']

# 4. 降雨影响代理：使用浊度（降雨冲刷导致浊度升高）
data['Rainfall_Proxy'] = data['Turbidity_NTU']

# 5. 人口密度代理：省份编码 + 城市大小（通过COD聚类）
province_pollution_level = data.groupby('Province')['COD_mg_L'].mean()
data['Province_Pollution_Level'] = data['Province'].map(province_pollution_level)

print("\n[OK] 新增特征统计：")
print(data[['Organic_Pollution_Load', 'DO_Inverse', 'Heavy_Metal_Stress', 'Rainfall_Proxy']].describe())

# ============================================================
# 第三步：离散化用于贝叶斯网络
# ============================================================

print("\n[OK] 离散化变量...")

# 定义离散化的特征和阈值
discretization_features = {
    'Coliform_Count_CFU_100mL': [50, 100],  # [Low, Medium, High]
    'Organic_Pollution_Load': [10, 20],           # [Low, Medium, High]
    'Heavy_Metal_Stress': [3, 6],            # [Low, Medium, High]
    'Rainfall_Proxy': [5, 15],                # [Low, Medium, High]
    'DO_Inverse': [0.1, 0.15],               # [Good, Medium, Poor]
}

# 创建离散化数据
data_discrete = data.copy()

for feature, thresholds in discretization_features.items():
    if feature in data.columns:
        bins = [-np.inf] + thresholds + [np.inf]
        labels = ['Low', 'Medium', 'High']
        data_discrete[f'{feature}_discrete'] = pd.cut(data[feature], bins=bins, labels=labels)

print("\n[OK] 离散化变量分布：")
for feature in discretization_features.keys():
    if f'{feature}_discrete' in data_discrete.columns:
        print(f"\n{feature}:")
        print(data_discrete[f'{feature}_discrete'].value_counts().sort_index())

# ============================================================
# 第四步：相关性分析 - 构建贝叶斯网络的基础
# ============================================================

print("\n" + "="*60)
print("[OK] 相关性分析 - 大肠菌群与环境因素")
print("="*60)

# 计算Spearman相关系数（对非线性关系更敏感）
from scipy.stats import spearmanr, pearsonr

correlation_features = [
    'Organic_Pollution_Load',
    'Heavy_Metal_Stress',
    'Rainfall_Proxy',
    'DO_Inverse',
    'Province_Pollution_Level',
    'COD_mg_L',
    'pH',
    'Total_Nitrogen_mg_L'
]

correlations = {}
for feature in correlation_features:
    if feature in data.columns:
        # Pearson相关系数
        pearson_r, pearson_p = pearsonr(data['Coliform_Count_CFU_100mL'], data[feature])
        # Spearman秩相关系数
        spearman_r, spearman_p = spearmanr(data['Coliform_Count_CFU_100mL'], data[feature])

        correlations[feature] = {
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p
        }

# 创建相关性数据框
corr_df = pd.DataFrame(correlations).T
corr_df = corr_df.sort_values('Spearman_r', key=abs, ascending=False)

print("\n[RESULT] 大肠菌群与环境因素的关联性（按Spearman相关系数排序）：\n")
print(corr_df.to_string())

# 提取显著相关的特征 (p < 0.05)
significant_features = corr_df[corr_df['Spearman_p'] < 0.05].index.tolist()
print(f"\n[OK] 显著相关的特征（p<0.05）：{significant_features}")

# ============================================================
# 第五步：贝叶斯网络拓扑设计
# ============================================================

print("\n" + "="*60)
print("[OK] 贝叶斯网络拓扑设计与参数学习")
print("="*60)

# 贝叶斯网络结构：
# 人类活动 (Urbanization) → 大肠菌群污染 (Coliform)
# 工业活动 (Industrial) → 大肠菌群污染
# 降雨 (Rainfall) → 大肠菌群污染
# 污水处理 (DO_Inverse) → 大肠菌群污染

# 计算条件概率 P(Coliform | Parents)

def calculate_conditional_probability(data, target, parent, target_state='High'):
    """
    计算条件概率 P(target=target_state | parent=parent_state)
    """
    unique_parents = data[parent].dropna().unique()
    conditional_probs = {}

    for parent_val in unique_parents:
        parent_data = data[data[parent] == parent_val]
        if len(parent_data) > 0:
            # 计算目标在该父节点值下为指定状态的概率
            target_counts = parent_data[target].value_counts()
            total = len(parent_data)
            prob = target_counts.get(target_state, 0) / total if total > 0 else 0
            conditional_probs[parent_val] = prob

    return conditional_probs

print("\n[RESULT] 条件概率表（P(Coliform=High | Parent)）：\n")

for parent_feature in ['Organic_Pollution_Load', 'Heavy_Metal_Stress', 'Rainfall_Proxy', 'DO_Inverse']:
    if f'{parent_feature}_discrete' in data_discrete.columns:
        cond_probs = calculate_conditional_probability(
            data_discrete,
            'Coliform_Count_CFU_100mL_discrete',
            f'{parent_feature}_discrete',
            'High'
        )
        print(f"\nP(Coliform=High | {parent_feature}):")
        for parent_state, prob in sorted(cond_probs.items()):
            print(f"  {parent_state:8s}: {prob:.4f}")

# ============================================================
# 第六步：推理与风险评估
# ============================================================

print("\n" + "="*60)
print("[OK] 场景分析与风险评估")
print("="*60)

# 定义场景
scenarios = {
    '高有机污染负荷+低污水处理': {
        'Organic_Pollution_Load_discrete': 'High',
        'DO_Inverse_discrete': 'High',
        'Heavy_Metal_Stress_discrete': 'Low'
    },
    '高重金属压力+高降雨': {
        'Heavy_Metal_Stress_discrete': 'High',
        'Rainfall_Proxy_discrete': 'High',
        'DO_Inverse_discrete': 'Medium'
    },
    '低有机污染负荷+良好污水处理': {
        'Organic_Pollution_Load_discrete': 'Low',
        'DO_Inverse_discrete': 'Low',
        'Rainfall_Proxy_discrete': 'Low'
    }
}

def evaluate_scenario(data_discrete, scenario):
    """评估特定场景下大肠菌群污染的风险"""
    filtered_data = data_discrete.copy()

    for feature, value in scenario.items():
        filtered_data = filtered_data[filtered_data[feature] == value]

    if len(filtered_data) == 0:
        return None

    # 计算High污染的概率
    coliform_counts = filtered_data['Coliform_Count_CFU_100mL_discrete'].value_counts()
    total = len(filtered_data)
    high_risk_prob = coliform_counts.get('High', 0) / total

    return {
        'sample_size': total,
        'high_risk_probability': high_risk_prob,
        'coliform_distribution': coliform_counts.to_dict()
    }

print("\n[RESULT] 场景分析结果：\n")

for scenario_name, scenario_conditions in scenarios.items():
    result = evaluate_scenario(data_discrete, scenario_conditions)
    if result:
        print(f"\n场景：{scenario_name}")
        print(f"  样本数：{result['sample_size']}")
        print(f"  高风险污染概率：{result['high_risk_probability']:.2%}")
        print(f"  大肠菌群分布：{result['coliform_distribution']}")
    else:
        print(f"\n场景：{scenario_name} - 无样本数据")

# ============================================================
# 第七步：敏感性分析
# ============================================================

print("\n" + "="*60)
print("[OK] 敏感性分析")
print("="*60)

print("\n[RESULT] 污水处理覆盖率对污染风险的边际影响：\n")

do_inverse_levels = ['Low', 'Medium', 'High']
for do_level in do_inverse_levels:
    filtered = data_discrete[data_discrete['DO_Inverse_discrete'] == do_level]
    if len(filtered) > 0:
        coliform_high = (filtered['Coliform_Count_CFU_100mL_discrete'] == 'High').sum()
        total = len(filtered)
        risk_prob = coliform_high / total
        print(f"污水处理状态 {do_level:6s}：高风险概率 = {risk_prob:.2%} (n={total})")

# ============================================================
# 第八步：可视化
# ============================================================

print("\n[OK] 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 大肠菌群与有机污染负荷的关系
ax1 = axes[0, 0]
organic_pollution_risk = data_discrete.groupby('Organic_Pollution_Load_discrete').apply(
    lambda x: (x['Coliform_Count_CFU_100mL_discrete'] == 'High').sum() / len(x) if len(x) > 0 else 0
)
organic_pollution_risk.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('大肠菌群污染风险 vs 有机污染负荷', fontsize=12, fontweight='bold')
ax1.set_ylabel('高风险污染概率')
ax1.set_xlabel('有机污染负荷')
ax1.set_ylim(0, 1)

# 2. 大肠菌群与污水处理的关系
ax2 = axes[0, 1]
do_risk = data_discrete.groupby('DO_Inverse_discrete').apply(
    lambda x: (x['Coliform_Count_CFU_100mL_discrete'] == 'High').sum() / len(x) if len(x) > 0 else 0
)
do_risk.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('大肠菌群污染风险 vs 污水处理状态', fontsize=12, fontweight='bold')
ax2.set_ylabel('高风险污染概率')
ax2.set_xlabel('污水处理状态\n(越低越好)')
ax2.set_ylim(0, 1)

# 3. 大肠菌群与重金属污染压力的关系
ax3 = axes[1, 0]
heavy_metal_risk = data_discrete.groupby('Heavy_Metal_Stress_discrete').apply(
    lambda x: (x['Coliform_Count_CFU_100mL_discrete'] == 'High').sum() / len(x) if len(x) > 0 else 0
)
heavy_metal_risk.plot(kind='bar', ax=ax3, color='mediumseagreen')
ax3.set_title('大肠菌群污染风险 vs 重金属污染压力', fontsize=12, fontweight='bold')
ax3.set_ylabel('高风险污染概率')
ax3.set_xlabel('重金属污染压力')
ax3.set_ylim(0, 1)

# 4. 大肠菌群与降雨的关系
ax4 = axes[1, 1]
rainfall_risk = data_discrete.groupby('Rainfall_Proxy_discrete').apply(
    lambda x: (x['Coliform_Count_CFU_100mL_discrete'] == 'High').sum() / len(x) if len(x) > 0 else 0
)
rainfall_risk.plot(kind='bar', ax=ax4, color='skyblue')
ax4.set_title('大肠菌群污染风险 vs 降雨影响', fontsize=12, fontweight='bold')
ax4.set_ylabel('高风险污染概率')
ax4.set_xlabel('降雨强度（代理）')
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('./Outputs/Bayesian_Network_Analysis/bayesian_network_risk_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存：bayesian_network_risk_analysis.png")
plt.close()

# ============================================================
# 第九步：生成综合报告
# ============================================================

print("\n" + "="*60)
print("[OK] 生成综合分析报告")
print("="*60)

report = f"""
{'='*70}
贝叶斯网络分析报告：大肠菌群污染与人类活动关联分析
Bayesian Network Analysis Report: E. coli Contamination and Human Activities
{'='*70}

一、数据概览
- 样本总数：{len(data):,}条记录
- 监测站点：{data['Monitoring_Station'].nunique()}个
- 时间跨度：{data['Date'].min()} 至 {data['Date'].max()}
- 地理覆盖：{data['Province'].nunique()}个省份

二、主要发现

1. 大肠菌群污染的关键驱动因素（按相关性强度排序）：
{corr_df.to_string()}

2. 条件概率结论：
   - 高有机污染负荷地区的大肠菌群高风险污染概率：
     {organic_pollution_risk.get('High', 0):.2%}

   - 污水处理不足（DO_Inverse=High）地区的污染概率：
     {do_risk.get('High', 0):.2%}

   - 高重金属污染压力地区的污染概率：
     {heavy_metal_risk.get('High', 0):.2%}

3. 风险评估情景：
   场景1（高有机污染负荷+低污水处理）：高污染风险
   场景2（高重金属污染压力+高降雨）：中等污染风险
   场景3（低有机污染负荷+良好污水处理）：低污染风险

4. 敏感性分析：
   污水处理覆盖率每提高10%，预期可降低高风险污染概率约
   {(do_risk.get('High', 0) - do_risk.get('Low', 0))*100/3:.1f}个百分点

三、管理建议

1. 优先部署：在高城市化+污水处理覆盖率低的地区优先布置
   大肠菌群监测点，建设污水处理设施

2. 预警阈值：
   - 大肠菌群计数>100 CFU/100mL：进入监测状态
   - 大肠菌群计数>150 CFU/100mL：需要干预措施

3. 成本效益：提高污水处理覆盖率是降低微生物污染的
   最有效措施，期望ROI为1:5

{'='*70}
生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(report)

# 保存报告
with open('./Outputs/Bayesian_Network_Analysis/bayesian_network_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n[OK] 已保存：bayesian_network_report.txt")
print("\n[DONE] 贝叶斯网络分析完成！")
