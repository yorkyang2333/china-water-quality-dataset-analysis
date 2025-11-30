"""
贝叶斯网络分析脚本 (Bayesian Network Analysis)
学习水质指标间的因果关系网络

功能：
1. 数据驱动结构学习（Hill-Climbing算法）
2. 专家知识网络构建
3. 网络结构可视化
4. 概率推理示例

日期: 2025-11-14
版本: v1.0

依赖: pip install pgmpy
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys

print("\n" + "="*80)
print("  贝叶斯网络分析")
print("="*80)

# 检查pgmpy库
PGMPY_AVAILABLE = False
try:
    from pgmpy.estimators import HillClimbSearch, BIC, K2
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
    print("[OK] pgmpy库已安装 (v1.0.0+)")
except ImportError as e:
    print(f"[ERROR] 缺少pgmpy库: {e}")
    print("[INFO] 请运行: pip install pgmpy")
    print("[INFO] 将生成简化版专家知识网络...")

# 自动检测数据集文件位置
def find_data_file(filename):
    """自动查找数据文件，支持多个可能的位置"""
    possible_paths = [
        filename,  # 当前目录
        f"../{filename}",  # 上级目录
        f"../../{filename}",  # 上上级目录
        os.path.join(os.path.dirname(__file__), "..", filename),  # 脚本所在目录的上级
        os.path.join(os.getcwd(), filename),  # 工作目录
        os.path.join(os.getcwd(), "..", filename),  # 工作目录的上级
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    return None

# 创建输出目录
OUTPUT_DIR = "Outputs/Bayesian_Network_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载数据
print("\n[STEP 1] 加载数据...")

data_file = find_data_file("preprocessed_data.csv")
if data_file is None:
    print("[ERROR] 找不到 preprocessed_data.csv 文件")
    print("[INFO] 请确保在项目根目录或scripts目录下运行")
    print(f"[INFO] 当前工作目录: {os.getcwd()}")
    sys.exit(1)

print(f"[OK] 找到数据文件: {data_file}")
data = pd.read_csv(data_file)

# 选择核心水质指标（简化为12个，便于可视化）
INDICATORS = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Nitrate_mg_L', 'Ammonia_N_mg_L',
    'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L',
    'COD_mg_L', 'BOD_mg_L', 'Turbidity_NTU', 'Water_Quality_Index'
]

# 离散化数据（贝叶斯网络需要离散数据或连续高斯）
print("\n[STEP 2] 数据离散化（3个等级：低/中/高）...")

data_discrete = data[INDICATORS].copy()
for col in INDICATORS:
    # 使用分位数进行离散化
    data_discrete[col] = pd.qcut(data[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

print(f"[OK] 离散化完成: {len(data_discrete)} 条记录")


# 2. 专家知识网络（基于物理化学原理）
print("\n[STEP 3] 构建专家知识网络...")

# 定义专家知识的因果关系（有向边）
expert_edges = [
    # 温度影响链
    ('Water_Temperature_C', 'Dissolved_Oxygen_mg_L'),  # 温度↑ → 溶解氧↓
    ('Water_Temperature_C', 'pH'),                      # 温度影响pH

    # 氮循环
    ('Ammonia_N_mg_L', 'Nitrate_mg_L'),                # 氨氮→硝酸盐（硝化作用）
    ('Ammonia_N_mg_L', 'Total_Nitrogen_mg_L'),         # 氨氮→总氮
    ('Nitrate_mg_L', 'Total_Nitrogen_mg_L'),           # 硝酸盐→总氮

    # 有机物降解
    ('COD_mg_L', 'BOD_mg_L'),                          # COD包含BOD
    ('BOD_mg_L', 'Dissolved_Oxygen_mg_L'),             # BOD消耗溶解氧
    ('COD_mg_L', 'Dissolved_Oxygen_mg_L'),             # COD消耗溶解氧

    # 磷循环
    ('Total_Phosphorus_mg_L', 'pH'),                   # 磷酸盐影响pH

    # 浊度影响
    ('Turbidity_NTU', 'Conductivity_uS_cm'),           # 浊度与电导率相关
    ('COD_mg_L', 'Turbidity_NTU'),                     # 有机物增加浊度

    # 综合水质指数
    ('pH', 'Water_Quality_Index'),
    ('Dissolved_Oxygen_mg_L', 'Water_Quality_Index'),
    ('COD_mg_L', 'Water_Quality_Index'),
    ('Ammonia_N_mg_L', 'Water_Quality_Index'),
    ('Total_Nitrogen_mg_L', 'Water_Quality_Index'),
    ('Total_Phosphorus_mg_L', 'Water_Quality_Index'),
]

# 创建专家知识网络
if PGMPY_AVAILABLE:
    expert_model = DiscreteBayesianNetwork(expert_edges)
    print(f"[OK] 专家知识网络已构建: {len(expert_edges)} 条边")
    print(f"    节点数: {len(expert_model.nodes())}")
    print(f"    边数: {len(expert_model.edges())}")
else:
    # 如果pgmpy不可用，使用NetworkX表示网络
    class SimpleNetwork:
        def __init__(self, edges):
            self._edges = edges
            self._nodes = set([n for edge in edges for n in edge])
        def edges(self):
            return self._edges
        def nodes(self):
            return self._nodes

    expert_model = SimpleNetwork(expert_edges)
    print(f"[OK] 专家知识网络已构建: {len(expert_edges)} 条边")
    print(f"    节点数: {len(expert_model.nodes())}")
    print(f"    边数: {len(expert_model.edges())}")


# 3. 数据驱动结构学习（如果pgmpy可用）
if PGMPY_AVAILABLE:
    print("\n[STEP 4] 数据驱动结构学习 (Hill-Climbing + BIC)...")

    try:
        # 使用前1000条数据加速学习（完整数据集太大）
        sample_data = data_discrete.dropna().head(5000)

        # Hill-Climbing搜索 + BIC评分
        hc = HillClimbSearch(sample_data)
        best_model = hc.estimate(scoring_method=BIC(sample_data),
                                max_iter=100,
                                max_indegree=3)  # 限制最大入度为3

        learned_model = DiscreteBayesianNetwork(best_model.edges())

        print(f"[OK] 数据驱动网络学习完成")
        print(f"    节点数: {len(learned_model.nodes())}")
        print(f"    边数: {len(learned_model.edges())}")

        # 保存学习到的边
        with open(f"{OUTPUT_DIR}/learned_edges.txt", 'w', encoding='utf-8') as f:
            f.write("数据驱动学习的因果关系:\n")
            f.write("="*60 + "\n\n")
            for edge in learned_model.edges():
                f.write(f"{edge[0]} → {edge[1]}\n")

        print(f"[OK] 学习到的边已保存: {OUTPUT_DIR}/learned_edges.txt")

    except Exception as e:
        print(f"[WARN] 结构学习失败: {e}")
        print("[INFO] 将仅使用专家知识网络")
        learned_model = None
else:
    learned_model = None
    print("\n[STEP 4] 跳过数据驱动学习（pgmpy未安装）")


# 4. 可视化网络结构
print("\n[STEP 5] 可视化网络结构...")

def visualize_bn(model, title, filename):
    """可视化贝叶斯网络"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # 创建NetworkX图
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # 节点分组着色（按类别）
    node_colors = []
    node_groups = {
        'temperature': ['Water_Temperature_C'],
        'oxygen': ['Dissolved_Oxygen_mg_L'],
        'pH': ['pH'],
        'nitrogen': ['Ammonia_N_mg_L', 'Nitrate_mg_L', 'Total_Nitrogen_mg_L'],
        'phosphorus': ['Total_Phosphorus_mg_L'],
        'organics': ['COD_mg_L', 'BOD_mg_L'],
        'physical': ['Conductivity_uS_cm', 'Turbidity_NTU'],
        '综合': ['Water_Quality_Index']
    }

    color_map = {
        'temperature': '#E74C3C',
        'oxygen': '#3498DB',
        'pH': '#9B59B6',
        'nitrogen': '#2ECC71',
        'phosphorus': '#F39C12',
        'organics': '#E67E22',
        'physical': '#95A5A6',
        '综合': '#34495E'
    }

    for node in G.nodes():
        for group, nodes in node_groups.items():
            if node in nodes:
                node_colors.append(color_map[group])
                break
        else:
            node_colors.append('#BDC3C7')

    # 使用层次化布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=2000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#7F8C8D', arrows=True,
                          arrowsize=20, arrowstyle='->', width=1.5,
                          connectionstyle='arc3,rad=0.1', ax=ax)

    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[group], label=group)
                      for group in color_map.keys()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 可视化专家知识网络
visualize_bn(expert_model,
            '专家知识贝叶斯网络（基于物理化学原理）',
            f"{OUTPUT_DIR}/expert_bayesian_network.png")
print(f"[OK] 专家网络可视化已保存: {OUTPUT_DIR}/expert_bayesian_network.png")

# 可视化数据驱动网络（如果有）
if learned_model is not None:
    visualize_bn(learned_model,
                '数据驱动贝叶斯网络（Hill-Climbing学习）',
                f"{OUTPUT_DIR}/learned_bayesian_network.png")
    print(f"[OK] 数据驱动网络可视化已保存: {OUTPUT_DIR}/learned_bayesian_network.png")


# 5. 网络对比分析
if learned_model is not None:
    print("\n[STEP 6] 网络对比分析...")

    expert_edges_set = set(expert_model.edges())
    learned_edges_set = set(learned_model.edges())

    # 计算重叠
    common_edges = expert_edges_set & learned_edges_set
    expert_only = expert_edges_set - learned_edges_set
    learned_only = learned_edges_set - expert_edges_set

    print(f"\n网络对比:")
    print(f"  专家知识边数: {len(expert_edges_set)}")
    print(f"  数据驱动边数: {len(learned_edges_set)}")
    print(f"  重叠边数: {len(common_edges)} ({len(common_edges)/len(expert_edges_set)*100:.1f}%)")
    print(f"  仅在专家网络: {len(expert_only)}")
    print(f"  仅在数据网络: {len(learned_only)}")

    # 保存对比结果
    with open(f"{OUTPUT_DIR}/network_comparison.txt", 'w', encoding='utf-8') as f:
        f.write("贝叶斯网络对比分析\n")
        f.write("="*60 + "\n\n")

        f.write(f"专家知识边数: {len(expert_edges_set)}\n")
        f.write(f"数据驱动边数: {len(learned_edges_set)}\n")
        f.write(f"重叠边数: {len(common_edges)}\n\n")

        f.write("重叠的因果关系:\n")
        f.write("-"*60 + "\n")
        for edge in sorted(common_edges):
            f.write(f"{edge[0]} → {edge[1]}\n")

        f.write("\n仅在专家网络中的关系:\n")
        f.write("-"*60 + "\n")
        for edge in sorted(expert_only):
            f.write(f"{edge[0]} → {edge[1]}\n")

        f.write("\n仅在数据网络中的关系:\n")
        f.write("-"*60 + "\n")
        for edge in sorted(learned_only):
            f.write(f"{edge[0]} → {edge[1]}\n")

    print(f"[OK] 对比结果已保存: {OUTPUT_DIR}/network_comparison.txt")


# 6. 生成分析报告
print("\n[STEP 7] 生成分析报告...")

with open(f"{OUTPUT_DIR}/bayesian_network_report.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("  贝叶斯网络分析报告\n")
    f.write("  Bayesian Network Analysis Report\n")
    f.write("="*80 + "\n\n")

    f.write("1. 专家知识网络\n")
    f.write("-"*80 + "\n")
    f.write(f"节点数: {len(expert_model.nodes())}\n")
    f.write(f"边数: {len(expert_model.edges())}\n\n")

    f.write("因果关系类别:\n")
    f.write("  - 温度影响链: 温度 → 溶解氧, pH\n")
    f.write("  - 氮循环: 氨氮 → 硝酸盐 → 总氮\n")
    f.write("  - 有机物降解: COD/BOD → 溶解氧\n")
    f.write("  - 磷循环: 总磷 → pH\n")
    f.write("  - 浊度影响: 浊度 ↔ 电导率, COD → 浊度\n")
    f.write("  - 综合指标: 多个指标 → 水质指数\n\n")

    if learned_model is not None:
        f.write("2. 数据驱动网络\n")
        f.write("-"*80 + "\n")
        f.write(f"节点数: {len(learned_model.nodes())}\n")
        f.write(f"边数: {len(learned_model.edges())}\n")
        f.write(f"学习方法: Hill-Climbing + BIC评分\n")
        f.write(f"数据量: 5000 条样本（离散化）\n\n")

        f.write("3. 网络对比\n")
        f.write("-"*80 + "\n")
        f.write(f"重叠度: {len(common_edges)/len(expert_edges_set)*100:.1f}%\n")
        f.write(f"专家知识得到验证的边: {len(common_edges)}/{len(expert_edges_set)}\n")
        f.write(f"数据发现的新关系: {len(learned_only)}\n\n")

    f.write("4. 应用价值\n")
    f.write("-"*80 + "\n")
    f.write("a) 理解水质指标间的因果机制\n")
    f.write("b) 识别关键污染源（父节点）\n")
    f.write("c) 预测干预效果（do-calculus）\n")
    f.write("d) 不确定性量化（概率推理）\n\n")

    f.write("5. 输出文件\n")
    f.write("-"*80 + "\n")
    f.write("- expert_bayesian_network.png: 专家知识网络可视化\n")
    if learned_model is not None:
        f.write("- learned_bayesian_network.png: 数据驱动网络可视化\n")
        f.write("- learned_edges.txt: 学习到的因果关系\n")
        f.write("- network_comparison.txt: 网络对比分析\n")
    f.write("- bayesian_network_report.txt: 本报告\n\n")

    f.write("="*80 + "\n")
    f.write("报告生成时间: 2025-11-14\n")
    f.write("="*80 + "\n")

print(f"[OK] 分析报告已保存: {OUTPUT_DIR}/bayesian_network_report.txt")

print("\n" + "="*80)
print("  贝叶斯网络分析完成")
print("="*80)
print(f"\n所有结果已保存至: {OUTPUT_DIR}/")
print("\n关键成果:")
print(f"  - 专家知识网络: {len(expert_model.edges())} 条因果关系")
if learned_model is not None:
    print(f"  - 数据驱动网络: {len(learned_model.edges())} 条因果关系")
    print(f"  - 重叠度: {len(common_edges)/len(expert_edges_set)*100:.1f}%")
else:
    print("  - 数据驱动网络: 未生成（pgmpy未安装）")
print("\n建议: 安装pgmpy以启用完整功能 (pip install pgmpy)")
