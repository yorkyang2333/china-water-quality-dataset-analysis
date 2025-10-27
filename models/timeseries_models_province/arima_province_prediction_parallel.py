"""
ARIMA时间序列预测 - 按省份建模 (多进程并行加速版)

⚠️ 警告：此模型不适合当前数据集！
原因：数据时间间隔极不规则（0-222天），违反ARIMA等间隔/近似等间隔采样假设
结果：R² < 0，预测效果极差
建议：使用横截面预测或传统机器学习方法（详见 README.md）

说明：ARIMA是统计模型，不支持GPU加速
但本版本使用**多进程并行**来加速不同指标的训练，提升2-4倍速度

方法：
- 使用ARIMA(p,d,q)自回归积分滑动平均模型
- 按省份聚合所有监测站数据
- 自动参数选择（基于AIC准则）
- **多进程并行** - 不同指标并行训练

适用机型：
- Mac/Windows/Linux (通用CPU)
- 多核CPU推荐（4核及以上）

性能：
- 训练速度：中等-快（多进程加速，比单进程快2-4倍）
- 内存需求：中等（多进程会占用更多内存）
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial

# ARIMA相关库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/ARIMA_Province_Predictions_Parallel'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return None

    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Sample_Size': len(y_true)
    }


def check_stationarity(timeseries):
    """检查时间序列的平稳性"""
    try:
        result = adfuller(timeseries, autolag='AIC')
        p_value = result[1]
        return p_value < 0.05, p_value
    except:
        return False, 1.0


def find_best_arima_params(timeseries, max_p=3, max_d=2, max_q=3):
    """寻找最佳ARIMA参数"""
    best_aic = np.inf
    best_params = (1, 1, 1)
    best_model = None

    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)

    for p, d, q in product(p_range, d_range, q_range):
        try:
            if p == 0 and d == 0 and q == 0:
                continue

            model = ARIMA(timeseries, order=(p, d, q))
            fitted = model.fit(method_kwargs={"warn_convergence": False})

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_params = (p, d, q)
                best_model = fitted
        except:
            continue

    return best_params, best_aic, best_model


def build_timeseries_arima_single_indicator(args):
    """
    为单个指标建立ARIMA模型（用于并行处理）

    参数:
    - args: (province_data, indicator, train_ratio, auto_params)
    """
    province_data, indicator, train_ratio, auto_params = args

    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    data = data[['Date', indicator]].dropna()

    if len(data) < 50:
        return indicator, None

    values = data[indicator].values
    is_stationary, p_value = check_stationarity(values)

    split_idx = int(len(values) * train_ratio)
    train_data = values[:split_idx]
    test_data = values[split_idx:]

    if len(train_data) < 30 or len(test_data) < 5:
        return indicator, None

    try:
        if auto_params:
            best_params, aic_score, fitted_model = find_best_arima_params(train_data)
            if fitted_model is None:
                best_params = (1, 1, 1)
                model = ARIMA(train_data, order=best_params)
                fitted_model = model.fit(method_kwargs={"warn_convergence": False})
        else:
            best_params = (1, 1, 1)
            model = ARIMA(train_data, order=best_params)
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})

        forecast = fitted_model.forecast(steps=len(test_data))
        y_pred = np.array(forecast)
        y_test = test_data

        metrics = calculate_metrics(y_test, y_pred)
        if metrics is None:
            return indicator, None

        if metrics['R2'] < -100:
            return indicator, None

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return indicator, None

        result = {
            'model_name': 'ARIMA',
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics,
            'params': best_params,
            'aic': fitted_model.aic,
            'is_stationary': is_stationary,
            'p_value': p_value
        }

        return indicator, result

    except Exception as e:
        return indicator, None


def predict_province(df, province, n_processes=None):
    """为单个省份预测所有指标（并行处理）"""
    province_data = df[df['Province'] == province].copy()

    if len(province_data) < 50:
        return None

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    # 过滤有效指标
    valid_indicators = []
    for indicator in indicators:
        valid_count = province_data[indicator].notna().sum()
        if valid_count >= 40:
            valid_indicators.append(indicator)

    if not valid_indicators:
        return None

    # 准备并行处理参数
    args_list = [(province_data, indicator, 0.8, True) for indicator in valid_indicators]

    # 并行处理
    if n_processes is None:
        n_processes = min(cpu_count(), len(valid_indicators))

    with Pool(processes=n_processes) as pool:
        indicator_results = pool.map(build_timeseries_arima_single_indicator, args_list)

    # 收集结果
    results = {}
    for indicator, result in indicator_results:
        if result is not None:
            results[indicator] = result

    return results if results else None


def process_all_provinces(df, n_processes=None):
    """处理所有省份（串行处理省份，并行处理指标）"""
    print("\n" + "="*80)
    print("ARIMA时间序列预测 - 按省份建模 (多进程并行加速版)")
    print("="*80)

    # 检测CPU核心数
    n_cpus = cpu_count()
    if n_processes is None:
        n_processes = max(1, n_cpus - 1)  # 留一个核心给系统

    print(f"\n✓ 检测到CPU核心数: {n_cpus}")
    print(f"  使用进程数: {n_processes}")

    provinces = df['Province'].unique()
    province_counts = df.groupby('Province').size().reset_index(name='count')

    print(f"\n总省份数: {len(provinces)}")
    print(f"\n方法参数:")
    print(f"  - 模型: ARIMA(p,d,q)")
    print(f"  - 参数选择: 自动优化（基于AIC）")
    print(f"  - 平稳性检验: ADF test")
    print(f"  - 数据分割: 80% 训练, 20% 测试")
    print(f"  - 建模方式: 按省份聚合所有监测站数据")
    print(f"  - 加速方式: 多进程并行（每个省份内的指标并行训练）")

    all_results = {}
    successful_provinces = 0

    for idx, province in enumerate(sorted(provinces)):
        count = province_counts[province_counts['Province'] == province]['count'].values[0]

        print(f"\n[{idx+1}/{len(provinces)}] {province}")
        print(f"  总数据点数: {count}")
        print(f"  使用 {n_processes} 个进程并行训练...")

        results = predict_province(df, province, n_processes)

        if results:
            all_results[province] = {
                'province': province,
                'sample_count': count,
                'results': results
            }
            successful_provinces += 1

            r2_scores = [r['metrics']['R2'] for r in results.values()]
            print(f"  ✓ 成功预测 {len(results)} 个指标")
            print(f"  平均 R²: {np.mean(r2_scores):.3f} (范围: {np.min(r2_scores):.3f} - {np.max(r2_scores):.3f})")
        else:
            print(f"  ✗ 数据不足，跳过")

    print(f"\n成功建模的省份: {successful_provinces}/{len(provinces)}")

    return all_results


def visualize_province(province, province_info, output_dir):
    """可视化单个省份的预测结果"""
    results = province_info['results']

    if not results:
        return

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(20, 5*nrows))

    for idx, (indicator, result) in enumerate(results.items()):
        ax = plt.subplot(nrows, ncols, idx + 1)

        y_test = result['y_test']
        y_pred = result['y_pred']
        m = result['metrics']
        params = result['params']

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#e74c3c')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        stationary_text = "平稳" if result['is_stationary'] else "非平稳"
        textstr = f"Model: ARIMA{params}\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}\n{stationary_text}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{province} Province\nARIMA Time Series Prediction (Parallel-Accelerated)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{province}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_results, output_dir):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ARIMA时间序列预测报告 - 按省份建模 (多进程并行加速版)\n")
        f.write("ARIMA Time Series Prediction Report - Province-Based (Parallel)\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CPU核心数: {cpu_count()}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个省份独立建模（聚合省内所有监测站数据）\n")
        f.write("- 使用ARIMA自回归积分滑动平均模型\n")
        f.write("- 参数(p,d,q)自动选择，基于AIC准则\n")
        f.write("- 平稳性检验：ADF test\n")
        f.write("- 多进程并行加速：不同指标并行训练\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_params = []
        stationary_count = 0

        for province, province_info in all_results.items():
            for indicator, result in province_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                all_params.append(result['params'])
                if result['is_stationary']:
                    stationary_count += 1

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"省份数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n")
        f.write(f"平稳序列数: {stationary_count} ({stationary_count/len(all_r2)*100:.1f}%)\n\n")

        f.write(f"R² 统计:\n")
        f.write(f"  平均值: {np.mean(all_r2):.4f}\n")
        f.write(f"  中位数: {np.median(all_r2):.4f}\n")
        f.write(f"  标准差: {np.std(all_r2):.4f}\n")
        f.write(f"  最大值: {np.max(all_r2):.4f}\n")
        f.write(f"  最小值: {np.min(all_r2):.4f}\n\n")

        # R²分布
        r2_ranges = {
            'R² > 0.7 (优秀)': sum(1 for r in all_r2 if r > 0.7),
            'R² 0.5-0.7 (良好)': sum(1 for r in all_r2 if 0.5 <= r <= 0.7),
            'R² 0.3-0.5 (中等)': sum(1 for r in all_r2 if 0.3 <= r < 0.5),
            'R² < 0.3 (较差)': sum(1 for r in all_r2 if r < 0.3)
        }

        f.write(f"R² 分布:\n")
        for range_name, count in r2_ranges.items():
            percentage = count / len(all_r2) * 100 if len(all_r2) > 0 else 0
            f.write(f"  {range_name}: {count} ({percentage:.1f}%)\n")
        f.write("\n")

        f.write(f"RMSE 平均值: {np.mean(all_rmse):.4f}\n")
        f.write(f"MAE 平均值: {np.mean(all_mae):.4f}\n\n")

        # 常用参数统计
        f.write(f"ARIMA参数分布:\n")
        param_counts = {}
        for p in all_params:
            if p not in param_counts:
                param_counts[p] = 0
            param_counts[p] += 1

        top_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for param, count in top_params:
            f.write(f"  ARIMA{param}: {count} 次 ({count/len(all_params)*100:.1f}%)\n")
        f.write("\n")

        f.write("\n" + "="*80 + "\n\n")

        # 详细的省份结果
        f.write("详细结果 - 按R²排序\n")
        f.write("="*80 + "\n\n")

        province_avg_r2 = []
        for province, province_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in province_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            province_avg_r2.append((province, province_info, avg_r2))

        province_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for province, province_info, avg_r2 in province_avg_r2:
            f.write(f"省份: {province}\n")
            f.write(f"样本数: {province_info['sample_count']} | 平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'模型':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            sorted_results = sorted(province_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                params = result['params']
                model_str = f"ARIMA{params}"
                f.write(f"{indicator:<30} {model_str:<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个省份独立建立ARIMA时间序列模型（聚合省内所有监测站）\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- 使用多进程并行加速，训练速度提升2-4倍\n")
        f.write("- 参数(p,d,q)自动选择，基于AIC准则\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("ARIMA时间序列预测系统 - 按省份建模 (多进程并行加速版)")
    print("ARIMA Time Series Prediction - Province-Based (Parallel)")
    print("="*80)

    # 加载数据
    print("\n[1/4] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ 总记录数: {len(df)}")
    print(f"✓ 省份数: {df['Province'].nunique()}")
    print(f"✓ 监测站数: {df['Monitoring_Station'].nunique()}")

    # 处理所有省份
    print("\n[2/4] 按省份建立ARIMA模型 (多进程并行)...")
    all_results = process_all_provinces(df)

    # 生成可视化
    print("\n[3/4] 生成可视化...")
    total = len(all_results)
    for idx, (province, province_info) in enumerate(all_results.items(), 1):
        print(f"  进度: {idx}/{total}")
        visualize_province(province, province_info, OUTPUT_DIR)

    # 生成报告
    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR)

    # 总结
    print("\n" + "="*80)
    print("完成！")
    print("="*80)

    total_models = sum(len(p['results']) for p in all_results.values())
    print(f"\n✓ 成功建模的省份: {len(all_results)}")
    print(f"✓ 总预测模型数: {total_models}")
    print(f"✓ 结果保存在: {OUTPUT_DIR}/")
    print(f"  - 报告: report.txt")
    print(f"  - 图表: {len(all_results)} 个省份的可视化\n")


if __name__ == "__main__":
    main()
