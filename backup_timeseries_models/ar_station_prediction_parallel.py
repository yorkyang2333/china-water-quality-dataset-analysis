"""
AR自回归模型时间序列预测 - 多进程并行加速版

方法：
- 使用AR(p)自回归模型
- 比ARIMA更简单，只考虑历史值的线性组合
- 多进程并行处理，充分利用多核CPU
- 自动选择最优滞后阶数p

适用机型：
- Mac/Windows/Linux (多核CPU)
- 推荐4核以上CPU以获得显著加速
- 无需GPU

性能：
- 训练速度：快（多进程并行）
- 内存需求：低（1-2GB）
- 相比单进程版本可提速2-4倍
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
from multiprocessing import Pool, cpu_count

# AR模型库
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/AR_Predictions_Parallel'
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


def find_best_ar_lag(timeseries, maxlag=10):
    """寻找最优滞后阶数"""
    best_aic = np.inf
    best_lag = 1

    max_possible_lag = min(maxlag, len(timeseries) // 3)

    for lag in range(1, max_possible_lag + 1):
        try:
            model = AutoReg(timeseries, lags=lag, old_names=False)
            fitted = model.fit()

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_lag = lag
        except:
            continue

    return best_lag, best_aic


def build_timeseries_ar(station_data, indicator, train_ratio=0.8):
    """使用AR模型进行时间序列预测"""
    data = station_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    data = data[['Date', indicator]].dropna()

    if len(data) < 25:
        return None

    values = data[indicator].values
    is_stationary, p_value = check_stationarity(values)

    split_idx = int(len(values) * train_ratio)
    train_data = values[:split_idx]
    test_data = values[split_idx:]

    if len(train_data) < 15 or len(test_data) < 2:
        return None

    try:
        best_lag, aic = find_best_ar_lag(train_data, maxlag=min(10, len(train_data) // 4))

        model = AutoReg(train_data, lags=best_lag, old_names=False)
        fitted_model = model.fit()

        forecast = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        y_pred = np.array(forecast)
        y_test = test_data

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return None

        metrics = calculate_metrics(y_test, y_pred)
        if metrics is None:
            return None

        if metrics['R2'] < -100:
            return None

        return {
            'model_name': 'AR',
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics,
            'lag': best_lag,
            'aic': fitted_model.aic,
            'is_stationary': is_stationary,
            'p_value': p_value
        }

    except Exception as e:
        return None


def predict_single_indicator(args):
    """预测单个指标（用于并行处理）"""
    station_data, indicator = args

    valid_count = station_data[indicator].notna().sum()
    if valid_count < 20:
        return indicator, None

    result = build_timeseries_ar(station_data, indicator)
    return indicator, result


def predict_station(df, province, city, station, n_processes=None):
    """为单个监测站预测所有指标（并行版本）"""
    station_data = df[
        (df['Province'] == province) &
        (df['City'] == city) &
        (df['Monitoring_Station'] == station)
    ].copy()

    if len(station_data) < 25:
        return None

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    tasks = [(station_data, ind) for ind in indicators]

    if n_processes is None:
        n_processes = min(cpu_count(), len(indicators))

    results = {}
    with Pool(processes=n_processes) as pool:
        parallel_results = pool.map(predict_single_indicator, tasks)

    for indicator, result in parallel_results:
        if result:
            results[indicator] = result

    return results if results else None


def process_all_stations(df, n_processes=None):
    """处理所有监测站"""
    print("\n" + "="*80)
    print("AR自回归模型时间序列预测 - 按监测站建模 (多进程并行版)")
    print("="*80)

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print(f"\nCPU核心数: {cpu_count()}")
    print(f"使用进程数: {n_processes}")

    stations_info = df.groupby(['Province', 'City', 'Monitoring_Station']).size().reset_index(name='count')
    stations_info = stations_info[stations_info['count'] >= 25]

    print(f"\n总监测站数: {len(stations_info)}")
    print(f"样本数>=25的监测站: {len(stations_info)}")
    print(f"\n方法参数:")
    print(f"  - 模型: AR(p) 自回归模型")
    print(f"  - 滞后阶数: 自动选择（基于AIC）")
    print(f"  - 平稳性检验: ADF test")
    print(f"  - 数据分割: 80% 训练, 20% 测试")
    print(f"  - 并行加速: {n_processes} 进程")

    all_results = {}
    successful_stations = 0

    for idx, row in stations_info.iterrows():
        province = row['Province']
        city = row['City']
        station = row['Monitoring_Station']

        print(f"\n[{idx+1}/{len(stations_info)}] {province} - {city} - {station}")
        print(f"  样本数: {row['count']}")

        results = predict_station(df, province, city, station, n_processes)

        if results:
            station_key = f"{province}_{city}_{station}"
            all_results[station_key] = {
                'province': province,
                'city': city,
                'station': station,
                'sample_count': row['count'],
                'results': results
            }
            successful_stations += 1

            r2_scores = [r['metrics']['R2'] for r in results.values()]
            print(f"  ✓ 成功预测 {len(results)} 个指标")
            print(f"  平均 R²: {np.mean(r2_scores):.3f} (范围: {np.min(r2_scores):.3f} - {np.max(r2_scores):.3f})")
        else:
            print(f"  ✗ 数据不足，跳过")

    print(f"\n成功建模的监测站: {successful_stations}/{len(stations_info)}")

    return all_results


def visualize_station(station_key, station_info, output_dir):
    """可视化单个监测站的预测结果"""
    results = station_info['results']

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
        lag = result['lag']

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#c0392b')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        stationary_text = "平稳" if result['is_stationary'] else "非平稳"
        textstr = f"Model: AR({lag})\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}\n{stationary_text}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='mistyrose', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{station_info['province']} - {station_info['city']} - {station_info['station']}\nAR Time Series Prediction (Parallel)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{station_key}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_results, output_dir):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AR自回归模型时间序列预测报告 - 按监测站建模 (多进程并行版)\n")
        f.write("AR Time Series Prediction Report - Parallel Processing\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个监测站独立建模\n")
        f.write("- 使用AR(p)自回归模型\n")
        f.write("- 滞后阶数p自动选择，基于AIC准则\n")
        f.write("- 平稳性检验：ADF test\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n")
        f.write(f"- 多进程并行处理，充分利用CPU资源\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_lags = []
        stationary_count = 0

        for station_key, station_info in all_results.items():
            for indicator, result in station_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                all_lags.append(result['lag'])
                if result['is_stationary']:
                    stationary_count += 1

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"监测站数量: {len(all_results)}\n")
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

        # 滞后阶数统计
        f.write(f"滞后阶数分布:\n")
        f.write(f"  平均滞后阶数: {np.mean(all_lags):.1f}\n")
        f.write(f"  最常用滞后阶数: {max(set(all_lags), key=all_lags.count)}\n\n")

        f.write("\n" + "="*80 + "\n\n")

        # 按省份汇总
        province_stats = {}
        for station_info in all_results.values():
            province = station_info['province']
            if province not in province_stats:
                province_stats[province] = []

            for result in station_info['results'].values():
                province_stats[province].append(result['metrics']['R2'])

        f.write("按省份统计\n")
        f.write("="*80 + "\n")
        for province in sorted(province_stats.keys()):
            r2_list = province_stats[province]
            f.write(f"\n{province}:\n")
            f.write(f"  监测站数: {sum(1 for v in all_results.values() if v['province'] == province)}\n")
            f.write(f"  模型数: {len(r2_list)}\n")
            f.write(f"  平均 R²: {np.mean(r2_list):.4f}\n")
            f.write(f"  最佳 R²: {np.max(r2_list):.4f}\n")

        f.write("\n" + "="*80 + "\n\n")

        # 详细的监测站结果
        f.write("详细结果 - 按R²排序\n")
        f.write("="*80 + "\n\n")

        station_avg_r2 = []
        for station_key, station_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in station_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            station_avg_r2.append((station_key, station_info, avg_r2))

        station_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for station_key, station_info, avg_r2 in station_avg_r2:
            f.write(f"监测站: {station_info['province']} - {station_info['city']} - {station_info['station']}\n")
            f.write(f"样本数: {station_info['sample_count']} | 平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'模型':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            sorted_results = sorted(station_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                model_str = f"AR({result['lag']})"
                f.write(f"{indicator:<30} {model_str:<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个监测站独立建立AR自回归模型\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- 使用多进程并行处理，提升运算速度\n")
        f.write("- AR模型简单高效，适合短期线性预测\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("AR自回归模型时间序列预测系统 - 按监测站建模 (多进程并行版)")
    print("AR Time Series Prediction - Parallel Processing")
    print("="*80)

    print("\n[1/4] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ 总记录数: {len(df)}")
    print(f"✓ 省份数: {df['Province'].nunique()}")
    print(f"✓ 监测站数: {df['Monitoring_Station'].nunique()}")

    print("\n[2/4] 按监测站建立AR模型（并行处理）...")
    all_results = process_all_stations(df)

    print("\n[3/4] 生成可视化...")
    total = len(all_results)
    for idx, (station_key, station_info) in enumerate(all_results.items(), 1):
        if idx % 10 == 0 or idx == total:
            print(f"  进度: {idx}/{total}")
        visualize_station(station_key, station_info, OUTPUT_DIR)

    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)

    total_models = sum(len(s['results']) for s in all_results.values())
    print(f"\n✓ 成功建模的监测站: {len(all_results)}")
    print(f"✓ 总预测模型数: {total_models}")
    print(f"✓ 结果保存在: {OUTPUT_DIR}/")
    print(f"  - 报告: report.txt")
    print(f"  - 图表: {len(all_results)} 个监测站的可视化\n")


if __name__ == "__main__":
    main()
