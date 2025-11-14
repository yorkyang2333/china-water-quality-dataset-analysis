"""
水质因子季节性特征分析 - SARIMA方法
Seasonality Analysis of Water Quality Factors using SARIMA

目标：
- 分析各个水质指标的季节性模式
- 使用SARIMA模型识别和量化季节性特征
- 生成季节性分解可视化图表
- 评估季节性强度和周期性

方法：
1. 数据重采样：将不规则时间序列重采样为月度数据（处理时间间隔不规则问题）
2. 季节性分解：使用STL或经典分解方法分解趋势、季节和残差分量
3. SARIMA建模：通过SARIMA参数推断季节性强度
4. ACF/PACF分析：自相关和偏自相关函数分析周期性
5. 季节性指数计算：量化季节性变化幅度

输出：
- 各因子的季节性分解图
- 季节性强度统计报告
- ACF/PACF图表
- 季节性模式对比图

适用范围：
- 按省份或按监测站聚合数据
- 至少需要2年以上的数据才能检测季节性

创建日期：2025-10-31
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# SARIMA和时间序列分析库
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats

warnings.filterwarnings('ignore')

# 设置中文字体

# 设置随机种子
np.random.seed(42)

# 输出目录
OUTPUT_DIR = 'Outputs/Seasonality_Analysis_SARIMA'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/decomposition', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/acf_pacf', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/seasonal_patterns', exist_ok=True)

# 水质指标列表
WATER_QUALITY_INDICATORS = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
    'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
    'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
    'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
    'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
]

def prepare_timeseries(df, indicator, groupby='Province', resample_freq='M'):
    """
    准备时间序列数据

    参数:
    - df: 原始数据
    - indicator: 水质指标
    - groupby: 聚合方式 ('Province' 或 'Monitoring_Station')
    - resample_freq: 重采样频率 ('M'=月, 'W'=周, 'D'=日)

    返回:
    - 重采样后的时间序列字典 {group_name: series}
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    timeseries_dict = {}

    groups = df[groupby].unique()

    for group in groups:
        group_data = df[df[groupby] == group].copy()

        # 检查数据量
        if len(group_data) < 30 or group_data[indicator].isna().sum() > len(group_data) * 0.5:
            continue

        # 设置日期索引
        group_data = group_data.set_index('Date')

        # 重采样为月度/周度数据（取平均值）
        resampled = group_data[indicator].resample(resample_freq).mean()

        # 去除缺失值并进行插值
        resampled = resampled.interpolate(method='linear', limit=3)
        resampled = resampled.dropna()

        # 至少需要24个月的数据才能检测季节性
        if len(resampled) >= 24:
            timeseries_dict[group] = resampled

    return timeseries_dict

def calculate_seasonality_strength(decomposition):
    """
    计算季节性强度指标

    公式: 1 - Var(Residual) / Var(Seasonal + Residual)
    值越接近1表示季节性越强
    """
    try:
        seasonal = decomposition.seasonal
        resid = decomposition.resid

        # 去除NaN
        seasonal = seasonal[~np.isnan(seasonal)]
        resid = resid[~np.isnan(resid)]

        if len(seasonal) == 0 or len(resid) == 0:
            return 0.0

        var_resid = np.var(resid)
        var_seasonal_resid = np.var(np.concatenate([seasonal, resid]))

        if var_seasonal_resid == 0:
            return 0.0

        strength = max(0, 1 - var_resid / var_seasonal_resid)
        return strength
    except:
        return 0.0

def calculate_trend_strength(decomposition):
    """
    计算趋势强度指标

    公式: 1 - Var(Residual) / Var(Trend + Residual)
    """
    try:
        trend = decomposition.trend
        resid = decomposition.resid

        # 去除NaN
        trend = trend[~np.isnan(trend)]
        resid = resid[~np.isnan(resid)]

        if len(trend) == 0 or len(resid) == 0:
            return 0.0

        var_resid = np.var(resid)
        var_trend_resid = np.var(np.concatenate([trend, resid]))

        if var_trend_resid == 0:
            return 0.0

        strength = max(0, 1 - var_resid / var_trend_resid)
        return strength
    except:
        return 0.0

def decompose_seasonality(timeseries, method='stl', period=12):
    """
    季节性分解

    参数:
    - timeseries: 时间序列数据
    - method: 'stl' 或 'additive' 或 'multiplicative'
    - period: 季节周期（月度数据用12）

    返回:
    - decomposition对象
    """
    try:
        if method == 'stl':
            # STL分解（更稳健）
            # STL的seasonal参数必须是奇数且>=3，这是平滑窗口大小，不是周期
            # 对于月度数据（周期12），通常使用seasonal=13或7
            # 对于季度数据（周期4），使用seasonal=7
            seasonal_window = 13 if period >= 12 else 7
            stl = STL(timeseries, seasonal=seasonal_window, period=period, robust=True)
            decomposition = stl.fit()
        else:
            # 经典分解
            decomposition = seasonal_decompose(
                timeseries,
                model=method,
                period=period,
                extrapolate_trend='freq'
            )

        return decomposition
    except Exception as e:
        print(f"  ⚠ 分解失败: {str(e)}")
        return None

def fit_sarima_for_seasonality(timeseries, period=12):
    """
    拟合SARIMA模型以评估季节性

    返回:
    - best_params: 最优参数
    - aic: AIC值
    - seasonal_params: 季节性参数 (P, D, Q)
    """
    try:
        best_aic = np.inf
        best_params = None

        # 简化的参数搜索空间
        p_range = [0, 1, 2]
        d_range = [0, 1]
        q_range = [0, 1, 2]

        P_range = [0, 1, 2]  # 季节性AR
        D_range = [0, 1]     # 季节性差分
        Q_range = [0, 1, 2]  # 季节性MA

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                try:
                                    model = SARIMAX(
                                        timeseries,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    fitted = model.fit(disp=False, maxiter=50)

                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_params = {
                                            'order': (p, d, q),
                                            'seasonal_order': (P, D, Q, period),
                                            'aic': fitted.aic
                                        }
                                except:
                                    continue

        return best_params
    except:
        return None

def test_stationarity(timeseries):
    """
    平稳性检验（ADF和KPSS双检验）

    返回:
    - is_stationary: 是否平稳
    - adf_pvalue: ADF检验p值
    - kpss_pvalue: KPSS检验p值
    """
    try:
        # ADF检验（原假设：非平稳）
        adf_result = adfuller(timeseries, autolag='AIC')
        adf_pvalue = adf_result[1]

        # KPSS检验（原假设：平稳）
        kpss_result = kpss(timeseries, regression='c', nlags='auto')
        kpss_pvalue = kpss_result[1]

        # 双重判断：ADF拒绝非平稳 且 KPSS不拒绝平稳
        is_stationary = (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)

        return is_stationary, adf_pvalue, kpss_pvalue
    except:
        return False, 1.0, 0.0

def visualize_decomposition(timeseries, decomposition, title, filename):
    """
    可视化季节性分解结果
    """
    if decomposition is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # 原始序列
    axes[0].plot(timeseries.index, timeseries.values, color='steelblue', linewidth=1.5)
    axes[0].set_ylabel('Original', fontsize=10)
    axes[0].set_title(f'Seasonal Decomposition: {title}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 趋势
    axes[1].plot(decomposition.trend.index, decomposition.trend.values,
                 color='orangered', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 季节性
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values,
                 color='green', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # 残差
    axes[3].plot(decomposition.resid.index, decomposition.resid.values,
                 color='purple', linewidth=1)
    axes[3].set_ylabel('Residual', fontsize=10)
    axes[3].set_xlabel('Date', fontsize=10)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_acf_pacf(timeseries, title, filename, lags=40):
    """
    可视化ACF和PACF图
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ACF图
    plot_acf(timeseries, lags=min(lags, len(timeseries)//2 - 1), ax=axes[0])
    axes[0].set_title(f'Autocorrelation Function (ACF): {title}', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # PACF图
    plot_pacf(timeseries, lags=min(lags, len(timeseries)//2 - 1), ax=axes[1])
    axes[1].set_title(f'Partial Autocorrelation Function (PACF): {title}', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_seasonal_pattern(seasonal_components, title, filename):
    """
    可视化多个组的季节性模式对比
    """
    if not seasonal_components:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for group_name, seasonal in seasonal_components.items():
        # 只取一个完整周期（12个月）
        if len(seasonal) >= 12:
            one_cycle = seasonal[:12]
            months = range(1, 13)
            ax.plot(months, one_cycle.values, marker='o', linewidth=2,
                   label=group_name, alpha=0.7)

    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Seasonal Component', fontsize=11)
    ax.set_title(f'Seasonal Pattern Comparison: {title}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_indicator_seasonality(df, indicator, groupby='Province'):
    """
    分析单个指标的季节性特征

    返回:
    - results: 包含各组季节性分析结果的字典
    """
    print(f"\n{'='*80}")
    print(f"分析指标: {indicator}")
    print(f"{'='*80}")

    # 准备时间序列数据（重采样为月度）
    print("  [1/5] 准备时间序列数据（月度重采样）...")
    timeseries_dict = prepare_timeseries(df, indicator, groupby, resample_freq='M')

    if not timeseries_dict:
        print(f"  ✗ 无足够数据，跳过")
        return None

    print(f"  ✓ 成功准备 {len(timeseries_dict)} 个时间序列")

    results = {}
    seasonal_components = {}

    for idx, (group_name, timeseries) in enumerate(timeseries_dict.items(), 1):
        print(f"\n  [{idx}/{len(timeseries_dict)}] {group_name}")
        print(f"    数据点数: {len(timeseries)}")

        # 平稳性检验
        is_stationary, adf_p, kpss_p = test_stationarity(timeseries)
        print(f"    平稳性: {'平稳' if is_stationary else '非平稳'} (ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f})")

        # 季节性分解（STL方法）
        print("    执行季节性分解 (STL)...")
        decomposition = decompose_seasonality(timeseries, method='stl', period=12)

        if decomposition is None:
            print("    ✗ 分解失败")
            continue

        # 计算季节性强度
        seasonality_strength = calculate_seasonality_strength(decomposition)
        trend_strength = calculate_trend_strength(decomposition)

        print(f"    季节性强度: {seasonality_strength:.3f}")
        print(f"    趋势强度: {trend_strength:.3f}")

        # SARIMA参数估计
        print("    拟合SARIMA模型...")
        sarima_params = fit_sarima_for_seasonality(timeseries, period=12)

        # 保存结果
        results[group_name] = {
            'timeseries': timeseries,
            'decomposition': decomposition,
            'seasonality_strength': seasonality_strength,
            'trend_strength': trend_strength,
            'is_stationary': is_stationary,
            'adf_pvalue': adf_p,
            'kpss_pvalue': kpss_p,
            'sarima_params': sarima_params,
            'data_length': len(timeseries)
        }

        seasonal_components[group_name] = decomposition.seasonal

        # 生成可视化
        if idx <= 10:  # 只为前10个组生成详细图表
            # 分解图
            viz_title = f"{indicator} - {group_name}"
            viz_filename = f"{OUTPUT_DIR}/decomposition/{indicator}_{group_name}_decomposition.png"
            visualize_decomposition(timeseries, decomposition, viz_title, viz_filename)

            # ACF/PACF图
            acf_filename = f"{OUTPUT_DIR}/acf_pacf/{indicator}_{group_name}_acf_pacf.png"
            visualize_acf_pacf(timeseries, viz_title, acf_filename, lags=24)

    # 生成季节性模式对比图
    print("\n  [5/5] 生成季节性模式对比图...")
    pattern_filename = f"{OUTPUT_DIR}/seasonal_patterns/{indicator}_seasonal_pattern_comparison.png"
    visualize_seasonal_pattern(seasonal_components, indicator, pattern_filename)

    return results

def generate_summary_report(all_results, output_dir):
    """
    生成综合季节性分析报告
    """
    report_path = f'{output_dir}/seasonality_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("水质因子季节性特征分析报告\n")
        f.write("Seasonality Analysis Report for Water Quality Factors\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 数据重采样：原始数据重采样为月度平均值（处理时间间隔不规则问题）\n")
        f.write("- 季节性分解：使用STL（Seasonal-Trend Decomposition using Loess）方法\n")
        f.write("- 平稳性检验：ADF和KPSS双重检验\n")
        f.write("- SARIMA建模：通过参数选择评估季节性模式\n")
        f.write("- 季节性强度：基于方差比计算（值越接近1表示季节性越强）\n")
        f.write("- ACF/PACF：自相关和偏自相关分析\n\n")

        # 汇总统计
        f.write("="*80 + "\n")
        f.write("总体统计\n")
        f.write("="*80 + "\n\n")

        total_indicators = len(all_results)
        f.write(f"分析的指标数: {total_indicators}\n\n")

        # 按指标统计
        f.write("各指标季节性强度统计：\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<35} {'平均季节性强度':>15} {'平均趋势强度':>15} {'样本数':>8}\n")
        f.write("-" * 80 + "\n")

        indicator_summary = []

        for indicator, results in all_results.items():
            if results is None:
                continue

            seasonality_strengths = [r['seasonality_strength'] for r in results.values()]
            trend_strengths = [r['trend_strength'] for r in results.values()]

            avg_seasonality = np.mean(seasonality_strengths)
            avg_trend = np.mean(trend_strengths)
            n_samples = len(results)

            indicator_summary.append({
                'indicator': indicator,
                'avg_seasonality': avg_seasonality,
                'avg_trend': avg_trend,
                'n_samples': n_samples
            })

        # 按季节性强度排序
        indicator_summary.sort(key=lambda x: x['avg_seasonality'], reverse=True)

        for item in indicator_summary:
            f.write(f"{item['indicator']:<35} {item['avg_seasonality']:>15.3f} "
                   f"{item['avg_trend']:>15.3f} {item['n_samples']:>8}\n")

        f.write("\n")

        # 季节性强度分类
        f.write("季节性强度分类：\n")
        f.write("-" * 80 + "\n")

        strong_seasonal = [i for i in indicator_summary if i['avg_seasonality'] > 0.6]
        moderate_seasonal = [i for i in indicator_summary if 0.3 <= i['avg_seasonality'] <= 0.6]
        weak_seasonal = [i for i in indicator_summary if i['avg_seasonality'] < 0.3]

        f.write(f"强季节性 (>0.6): {len(strong_seasonal)} 个指标\n")
        for item in strong_seasonal:
            f.write(f"  - {item['indicator']}: {item['avg_seasonality']:.3f}\n")

        f.write(f"\n中等季节性 (0.3-0.6): {len(moderate_seasonal)} 个指标\n")
        for item in moderate_seasonal:
            f.write(f"  - {item['indicator']}: {item['avg_seasonality']:.3f}\n")

        f.write(f"\n弱季节性 (<0.3): {len(weak_seasonal)} 个指标\n")
        for item in weak_seasonal:
            f.write(f"  - {item['indicator']}: {item['avg_seasonality']:.3f}\n")

        f.write("\n" + "="*80 + "\n\n")

        # 详细结果
        f.write("详细结果 - 按季节性强度排序\n")
        f.write("="*80 + "\n\n")

        for item in indicator_summary:
            indicator = item['indicator']
            results = all_results[indicator]

            if results is None:
                continue

            f.write(f"指标: {indicator}\n")
            f.write(f"平均季节性强度: {item['avg_seasonality']:.3f} | 平均趋势强度: {item['avg_trend']:.3f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'组别':<20} {'季节性强度':>12} {'趋势强度':>10} {'平稳性':>8} "
                   f"{'数据点':>8} {'SARIMA参数':>20}\n")
            f.write("-" * 80 + "\n")

            # 按季节性强度排序
            sorted_results = sorted(results.items(),
                                   key=lambda x: x[1]['seasonality_strength'],
                                   reverse=True)

            for group_name, result in sorted_results:
                stationary_text = "平稳" if result['is_stationary'] else "非平稳"

                sarima_str = "N/A"
                if result['sarima_params']:
                    order = result['sarima_params']['order']
                    s_order = result['sarima_params']['seasonal_order']
                    sarima_str = f"({order[0]},{order[1]},{order[2]})x({s_order[0]},{s_order[1]},{s_order[2]},{s_order[3]})"

                f.write(f"{group_name:<20} {result['seasonality_strength']:>12.3f} "
                       f"{result['trend_strength']:>10.3f} {stationary_text:>8} "
                       f"{result['data_length']:>8} {sarima_str:>20}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 季节性强度：值越接近1表示季节性模式越明显\n")
        f.write("- 趋势强度：值越接近1表示长期趋势越明显\n")
        f.write("- SARIMA参数：(p,d,q)x(P,D,Q,s) 其中P,D,Q为季节性参数\n")
        f.write("- 平稳性：通过ADF和KPSS双重检验判断\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")

def create_heatmap_seasonality(all_results, output_dir):
    """
    创建季节性强度热力图
    """
    # 准备数据
    indicators = []
    groups = set()

    for indicator, results in all_results.items():
        if results is None:
            continue
        indicators.append(indicator)
        for group_name in results.keys():
            groups.add(group_name)

    groups = sorted(list(groups))

    # 检查是否有足够数据
    if len(indicators) == 0 or len(groups) == 0:
        print("  ⚠ 没有足够的数据生成热力图，跳过")
        return

    # 创建矩阵
    matrix = np.zeros((len(indicators), len(groups)))

    for i, indicator in enumerate(indicators):
        results = all_results[indicator]
        for j, group in enumerate(groups):
            if group in results:
                matrix[i, j] = results[group]['seasonality_strength']
            else:
                matrix[i, j] = np.nan

    # 检查矩阵是否全为NaN
    if np.all(np.isnan(matrix)):
        print("  ⚠ 矩阵全为NaN，无法生成热力图，跳过")
        return

    # 检查是否有有效值
    valid_values = matrix[~np.isnan(matrix)]
    if len(valid_values) == 0:
        print("  ⚠ 没有有效的季节性强度值，无法生成热力图，跳过")
        return

    # 绘制热力图
    plt.figure(figsize=(max(12, len(groups)*0.5), max(8, len(indicators)*0.5)))

    sns.heatmap(matrix,
                xticklabels=groups,
                yticklabels=[ind.replace('_', ' ') for ind in indicators],
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Seasonality Strength'},
                linewidths=0.5,
                vmin=0,
                vmax=1)

    plt.title('Seasonality Strength Heatmap Across Indicators and Groups',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Group', fontsize=11)
    plt.ylabel('Water Quality Indicator', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = f'{output_dir}/seasonality_heatmap.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 热力图已保存: {filename}")

def main():
    """主函数"""
    print("="*80)
    print("水质因子季节性特征分析系统 - SARIMA方法")
    print("Seasonality Analysis System for Water Quality Factors - SARIMA Method")
    print("="*80)

    # 加载数据
    print("\n[1/5] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data_feature_eng.csv', encoding='utf-8')
    except:
        try:
            df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
        except:
            df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ 总记录数: {len(df)}")
    print(f"✓ 省份数: {df['Province'].nunique()}")
    print(f"✓ 监测站数: {df['Monitoring_Station'].nunique()}")

    # 分析各指标的季节性
    print("\n[2/5] 分析各指标的季节性特征...")
    print("注意：数据将重采样为月度平均值以处理时间间隔不规则问题")

    all_results = {}

    for idx, indicator in enumerate(WATER_QUALITY_INDICATORS, 1):
        print(f"\n进度: {idx}/{len(WATER_QUALITY_INDICATORS)}")

        results = analyze_indicator_seasonality(
            df,
            indicator,
            groupby='Province'  # 可改为 'Monitoring_Station'
        )

        if results:
            all_results[indicator] = results
            print(f"✓ 完成: {indicator}")
        else:
            print(f"✗ 跳过: {indicator} (数据不足)")

    # 生成季节性强度热力图
    print("\n[3/5] 生成季节性强度热力图...")
    create_heatmap_seasonality(all_results, OUTPUT_DIR)

    # 生成综合报告
    print("\n[4/5] 生成综合报告...")
    generate_summary_report(all_results, OUTPUT_DIR)

    # 总结
    print("\n[5/5] 完成！")
    print("\n" + "="*80)
    print("分析完成总结")
    print("="*80)

    print(f"\n✓ 成功分析的指标数: {len(all_results)}/{len(WATER_QUALITY_INDICATORS)}")
    print(f"✓ 结果保存在: {OUTPUT_DIR}/")
    print(f"  - 综合报告: seasonality_report.txt")
    print(f"  - 季节性分解图: decomposition/")
    print(f"  - ACF/PACF图: acf_pacf/")
    print(f"  - 季节性模式对比图: seasonal_patterns/")
    print(f"  - 季节性强度热力图: seasonality_heatmap.png")

    # 显示Top 5强季节性指标
    if all_results:
        print("\n" + "-"*80)
        print("Top 5 强季节性指标:")

        indicator_seasonality = []
        for indicator, results in all_results.items():
            avg_strength = np.mean([r['seasonality_strength'] for r in results.values()])
            indicator_seasonality.append((indicator, avg_strength))

        indicator_seasonality.sort(key=lambda x: x[1], reverse=True)

        for i, (indicator, strength) in enumerate(indicator_seasonality[:5], 1):
            print(f"  {i}. {indicator}: {strength:.3f}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
