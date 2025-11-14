"""
生成论文用图表
Generate figures for academic paper

用途: 生成PAPER_DELIVERY_SUMMARY.md中提到的缺失图表
- 图1: 数据诊断三联图 (采样间隔分布 / ACF图 / 2023 vs 2024密度对比)
- 图2: 完整方法对比柱状图 (时间序列 vs 横截面回归)

作者: Claude Code
日期: 2025-11-13
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# 配置
OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载数据"""
    print("\n[LOAD] 正在加载数据...")

    data_paths = [
        'preprocessed_data_feature_eng.csv',
        '../../preprocessed_data_feature_eng.csv',
        'preprocess/preprocessed_data_feature_eng.csv',
    ]

    for path in data_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            print(f"[OK] 从 {path} 加载数据")
            print(f"  数据大小: {data.shape[0]} 行, {data.shape[1]} 列")
            return data

    print("[ERROR] 找不到数据文件")
    return None

def generate_data_diagnosis_figure(data):
    """
    生成图1: 数据诊断三联图
    [A] 采样间隔分布
    [B] ACF图 (COD_mg_L为例)
    [C] 2023 vs 2024密度对比
    """
    print("\n[FIGURE 1] 正在生成数据诊断三联图...")

    # 处理日期
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    data = data.sort_values('Date')

    # 创建3x1子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # === [A] 采样间隔分布 ===
    print("  [A] 计算采样间隔分布...")

    # 按Province分组计算间隔
    all_gaps = []
    for province in data['Province'].unique():
        province_data = data[data['Province'] == province].copy()
        province_data = province_data.sort_values('Date')
        gaps = province_data['Date'].diff().dt.days.dropna()
        all_gaps.extend(gaps.values)

    all_gaps = np.array(all_gaps)
    all_gaps = all_gaps[all_gaps > 0]  # 移除0或负值

    # 计算统计量
    cv = all_gaps.std() / all_gaps.mean()

    # 绘制直方图
    axes[0].hist(all_gaps, bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[0].axvline(all_gaps.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'均值 = {all_gaps.mean():.2f}天')
    axes[0].axvline(np.median(all_gaps), color='green', linestyle='--', linewidth=2,
                    label=f'中位数 = {np.median(all_gaps):.2f}天')
    axes[0].set_xlabel('采样间隔 (天)', fontsize=12)
    axes[0].set_ylabel('密度', fontsize=12)
    axes[0].set_title(f'[A] 采样间隔分布\nCV = {cv:.2f} (阈值<0.2)',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 50)  # 限制x轴范围

    print(f"    CV = {cv:.4f}")
    print(f"    均值 = {all_gaps.mean():.2f}天")
    print(f"    中位数 = {np.median(all_gaps):.2f}天")

    # === [B] ACF图 (COD_mg_L) ===
    print("  [B] 计算ACF...")

    # 使用北京省的COD数据作为例子
    beijing_data = data[data['Province'] == 'Beijing'].copy()
    beijing_data = beijing_data.sort_values('Date')
    cod_values = beijing_data['COD_mg_L'].dropna().values

    # 计算ACF
    lags = 20
    acf_values = acf(cod_values, nlags=lags, fft=True)

    # 绘制ACF
    axes[1].bar(range(len(acf_values)), acf_values, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].axhline(y=0.3, color='red', linestyle='--', linewidth=2,
                    label='阈值 = 0.3')
    axes[1].axhline(y=-0.3, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('滞后阶数', fontsize=12)
    axes[1].set_ylabel('自相关系数 (ACF)', fontsize=12)
    axes[1].set_title(f'[B] 自相关函数 (ACF)\nCOD_mg_L, 北京省\nACF(lag=1) = {acf_values[1]:.4f}',
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(-0.5, 1.0)

    print(f"    ACF(lag=1) = {acf_values[1]:.4f}")

    # === [C] 2023 vs 2024密度对比 ===
    print("  [C] 计算年度密度对比...")

    data['Year'] = data['Date'].dt.year

    # 计算每年的样本数
    year_2023 = data[data['Year'] == 2023]
    year_2024 = data[data['Year'] == 2024]

    count_2023 = len(year_2023)
    count_2024 = len(year_2024)

    # 计算密度 (样本/天)
    days_2023 = (year_2023['Date'].max() - year_2023['Date'].min()).days + 1
    days_2024 = (year_2024['Date'].max() - year_2024['Date'].min()).days + 1

    density_2023 = count_2023 / days_2023
    density_2024 = count_2024 / days_2024

    # KS检验 (值域分布)
    cod_2023 = year_2023['COD_mg_L'].dropna()
    cod_2024 = year_2024['COD_mg_L'].dropna()
    ks_stat, ks_pval = stats.ks_2samp(cod_2023, cod_2024)

    # 绘制柱状图
    years = ['2023', '2024']
    densities = [density_2023, density_2024]
    colors = ['#2E86C1', '#E74C3C']

    bars = axes[2].bar(years, densities, alpha=0.7, edgecolor='black', color=colors)
    axes[2].set_ylabel('采样密度 (样本/天)', fontsize=12)
    axes[2].set_title(f'[C] 年度采样密度对比\n2024密度 = {density_2024/density_2023*100:.1f}% of 2023\nKS检验: D={ks_stat:.3f}, p={ks_pval:.3f}',
                     fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{density:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    print(f"    2023: {count_2023}样本, {density_2023:.2f}样本/天")
    print(f"    2024: {count_2024}样本, {density_2024:.2f}样本/天")
    print(f"    2024密度 = {density_2024/density_2023*100:.1f}% of 2023")
    print(f"    KS检验: D={ks_stat:.3f}, p={ks_pval:.3f}")

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/data_diagnosis_tri_panel.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[OK] 数据诊断三联图已保存: {save_path}")
    plt.close()

def generate_method_comparison_figure():
    """
    生成图2: 完整方法对比柱状图
    包含: Naive / ARIMA / LSTM / 混合 / XGBoost回归 / 分类
    """
    print("\n[FIGURE 2] 正在生成方法对比柱状图...")

    # 方法性能数据 (基于实验结果和PAPER_DELIVERY_SUMMARY.md)
    methods = [
        'Naive\n基线',
        'ARIMA\n时序',
        'LSTM\n时序',
        'LSTM+RF\n混合',
        'XGBoost\n回归\n(平均)',
        'XGBoost\n回归\n(成功4个)',
        'XGBoost\n分类\n(准确率)'
    ]

    # R²或等效指标
    r2_scores = [
        0.00,    # Naive基线
        -0.15,   # ARIMA
        -0.10,   # LSTM
        -0.30,   # LSTM+RF混合
        0.54,    # XGBoost回归(平均所有7个指标)
        0.97,    # XGBoost回归(平均成功的4个指标)
        0.9958   # XGBoost分类(准确率转换)
    ]

    colors = [
        '#95A5A6',  # Naive - 灰色
        '#E74C3C',  # ARIMA - 红色
        '#E67E22',  # LSTM - 橙色
        '#C0392B',  # 混合 - 深红
        '#F39C12',  # XGBoost回归平均 - 黄色
        '#27AE60',  # XGBoost回归成功 - 绿色
        '#2E86C1',  # XGBoost分类 - 蓝色
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制柱状图
    bars = ax.bar(range(len(methods)), r2_scores, alpha=0.8, edgecolor='black',
                   color=colors, width=0.7)

    # 基线和目标线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='基线 (R^2=0, Naive)')
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label='良好阈值 (R^2=0.5)')

    # 设置标签
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('R^2 分数 (或等效指标)', fontsize=13, fontweight='bold')
    ax.set_title('水质预测方法性能对比\n时间序列 vs 横截面方法',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        va = 'bottom' if score >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}' if abs(score) < 1.0 else f'{score:.4f}',
               ha='center', va=va, fontsize=11, fontweight='bold')

    # 添加注释
    ax.text(0.02, 0.98,
           '失败区域: R^2<0\n时间序列方法全部失败',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.7))

    ax.text(0.98, 0.98,
           '成功区域: R^2>0.5\n横截面方法显著优于时间序列',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='#D5F4E6', alpha=0.7))

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/method_comparison_complete.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[OK] 方法对比图已保存: {save_path}")
    plt.close()

    # 打印对比表格
    print("\n  方法性能对比表:")
    print("  " + "="*70)
    print(f"  {'方法':<30} {'R^2':<10} {'评价':<20}")
    print("  " + "-"*70)

    for method, score in zip(methods, r2_scores):
        method_clean = method.replace('\n', ' ')
        if score < 0:
            evaluation = '失败 (比Naive更差)'
        elif score < 0.5:
            evaluation = '较差'
        elif score < 0.8:
            evaluation = '良好'
        elif score < 0.95:
            evaluation = '优秀'
        else:
            evaluation = '卓越'

        print(f"  {method_clean:<30} {score:<10.4f} {evaluation:<20}")

    print("  " + "="*70)

def main():
    print("\n" + "="*70)
    print("  生成论文用图表")
    print("  Generate Figures for Academic Paper")
    print("="*70)

    # 加载数据
    data = load_data()
    if data is None:
        return

    # 生成图1: 数据诊断三联图
    generate_data_diagnosis_figure(data)

    # 生成图2: 方法对比柱状图
    generate_method_comparison_figure()

    print(f"\n[DONE] 完成! 所有图表已保存到: {OUTPUT_DIR}/")
    print("\n生成的图表:")
    print("  1. data_diagnosis_tri_panel.png - 数据诊断三联图")
    print("  2. method_comparison_complete.png - 完整方法对比图")

if __name__ == "__main__":
    main()
