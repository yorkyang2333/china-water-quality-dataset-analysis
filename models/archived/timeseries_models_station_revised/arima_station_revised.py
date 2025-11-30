"""
ARIMA时间序列预测 - 2023训练/2024验证版

⭐ 关键修改：
1. 仅使用2023年的数据进行模型训练
2. 2024年的数据作为验证/测试集
3. 可视化输出改为折线图（实际值vs预测值随时间变化）

方法：
- 使用ARIMA(p,d,q)自回归积分滑动平均模型
- 自动参数选择（基于AIC准则）
- 纯时间序列建模，每个监测站独立训练
- 时间序列分解 + 平稳性检验

适用机型：
- Mac/Windows/Linux (通用CPU)
- 无需GPU，适合笔记本电脑运行

日期：2025-10-25
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
from itertools import product

# ARIMA相关库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/ARIMA_2023Train_2024Valid_Station'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Station_Plots", exist_ok=True)


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
    """检查时间序列的平稳性（ADF检验）"""
    try:
        result = adfuller(timeseries, autolag='AIC')
        p_value = result[1]
        return p_value < 0.05, p_value
    except:
        return False, 1.0


def find_best_arima_params(timeseries, max_p=3, max_d=2, max_q=3):
    """寻找最佳ARIMA参数（基于AIC准则）"""
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


def build_timeseries_arima(station_data, indicator, auto_params=True):
    """
    ⭐ 修改：使用2023年数据训练，2024年数据测试
    """
    data = station_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    data = data[['Date', indicator]].dropna()

    if len(data) < 30:
        return None

    # ⭐ 添加年份列
    data['Year'] = data['Date'].dt.year

    # ⭐ 按年份分割（2023训练，2024测试）
    train_data = data[data['Year'] == 2023]
    test_data = data[data['Year'] == 2024]

    if len(train_data) < 15 or len(test_data) < 1:
        return None

    train_values = train_data[indicator].values
    test_values = test_data[indicator].values
    test_dates = test_data['Date'].values  # ⭐ 保存日期

    # 检查平稳性
    is_stationary, p_value = check_stationarity(train_values)

    try:
        # 选择ARIMA参数
        if auto_params:
            best_params, aic_score, fitted_model = find_best_arima_params(train_values)
            if fitted_model is None:
                best_params = (1, 1, 1)
                model = ARIMA(train_values, order=best_params)
                fitted_model = model.fit(method_kwargs={"warn_convergence": False})
        else:
            best_params = (1, 1, 1)
            model = ARIMA(train_values, order=best_params)
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})

        # 预测
        forecast = fitted_model.forecast(steps=len(test_values))
        y_pred = np.array(forecast)
        y_test = test_values

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred)
        if metrics is None:
            return None

        # 验证R²的合理性
        if metrics['R2'] < -100:
            return None

        # 检查预测值的合理性
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return None

        return {
            'model_name': 'ARIMA',
            'y_test': y_test,
            'y_pred': y_pred,
            'test_dates': test_dates,  # ⭐ 新增
            'train_size': len(train_values),
            'test_size': len(test_values),
            'metrics': metrics,
            'params': best_params,
            'aic': fitted_model.aic,
            'is_stationary': is_stationary,
            'p_value': p_value
        }

    except Exception:
        return None


def predict_station(df, province, city, station):
    """为单个监测站预测所有指标"""
    station_data = df[
        (df['Province'] == province) &
        (df['City'] == city) &
        (df['Monitoring_Station'] == station)
    ].copy()

    if len(station_data) < 30:
        return None

    # ⭐ 检查是否有2023和2024的数据
    station_data['Date'] = pd.to_datetime(station_data['Date'])
    has_2023 = (station_data['Date'].dt.year == 2023).any()
    has_2024 = (station_data['Date'].dt.year == 2024).any()

    if not (has_2023 and has_2024):
        return None

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    results = {}

    for indicator in indicators:
        valid_count = station_data[indicator].notna().sum()
        if valid_count < 25:
            continue

        result = build_timeseries_arima(station_data, indicator)
        if result:
            results[indicator] = result

    return results if results else None


def process_all_stations(df):
    """处理所有监测站"""
    print("\n" + "="*80)
    print("ARIMA时间序列预测 - 2023训练/2024验证 (Station级别)")
    print("="*80)

    stations_info = df.groupby(['Province', 'City', 'Monitoring_Station']).size().reset_index(name='count')
    stations_info = stations_info[stations_info['count'] >= 30]

    print(f"\n总监测站数: {len(stations_info)}")
    print(f"\n方法参数:")
    print(f"  - 模型: ARIMA(p,d,q)")
    print(f"  - 训练集: 2023年数据")
    print(f"  - 测试集: 2024年数据")
    print(f"  - 参数选择: 自动优化（基于AIC）")

    all_results = {}
    successful_stations = 0
    skipped_no_data = 0

    for idx, row in stations_info.iterrows():
        province = row['Province']
        city = row['City']
        station = row['Monitoring_Station']

        print(f"\n[{idx+1}/{len(stations_info)}] {province} - {city} - {station}")
        print(f"  样本数: {row['count']}")

        results = predict_station(df, province, city, station)

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
            print(f"  平均 R²: {np.mean(r2_scores):.3f}")
        else:
            skipped_no_data += 1
            print(f"  ✗ 数据不足或缺少2023/2024数据")

    print(f"\n成功: {successful_stations}, 跳过: {skipped_no_data}")
    return all_results


def visualize_station(station_key, station_info, output_dir):
    """
    ⭐ 可视化 - 使用折线图
    """
    results = station_info['results']
    if not results:
        return

    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))

    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (indicator, result) in enumerate(results.items()):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        y_test = result['y_test']
        y_pred = result['y_pred']
        test_dates = pd.to_datetime(result['test_dates'])
        m = result['metrics']
        params = result['params']

        # ⭐ 绘制折线图
        ax.plot(test_dates, y_test, marker='o', linewidth=2, markersize=5,
                color='steelblue', label='Actual (2024)', alpha=0.8)
        ax.plot(test_dates, y_pred, marker='s', linewidth=2, markersize=5,
                color='orangered', label='Predicted', alpha=0.8, linestyle='--')

        stationary_text = "平稳" if result['is_stationary'] else "非平稳"
        textstr = f"ARIMA{params} ({stationary_text})\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8))

        ax.set_title(indicator.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_xlabel('Date (2024)', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    # 隐藏多余的子图
    for idx in range(len(results), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    title = f"{station_info['province']} - {station_info['city']} - {station_info['station']}\nARIMA Prediction (Train: 2023, Test: 2024)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/Station_Plots/{station_key}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


def generate_report(all_results, output_dir):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ARIMA时间序列预测报告 - 2023训练/2024验证 (Station级别)\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("✓ 训练集：仅使用2023年的数据\n")
        f.write("✓ 测试集：2024年的数据\n")
        f.write("✓ 每个监测站独立建模\n")
        f.write("✓ 使用ARIMA自回归积分滑动平均模型\n")
        f.write("✓ 参数(p,d,q)自动选择，基于AIC准则\n")
        f.write("✓ 可视化：折线图显示实际值vs预测值\n\n")

        # 统计
        all_r2, all_rmse, all_mae = [], [], []
        all_params, stationary_count = [], 0
        train_sizes, test_sizes = [], []

        for station_info in all_results.values():
            for result in station_info['results'].values():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                all_params.append(result['params'])
                train_sizes.append(result['train_size'])
                test_sizes.append(result['test_size'])
                if result['is_stationary']:
                    stationary_count += 1

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"监测站数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n")
        if len(all_r2) > 0:
            f.write(f"平稳序列: {stationary_count} ({stationary_count/len(all_r2)*100:.1f}%)\n")
        else:
            f.write(f"平稳序列: {stationary_count} (无有效模型)\n")
        if train_sizes:
            f.write(f"平均训练集大小（2023）: {np.mean(train_sizes):.1f}\n")
        else:
            f.write("平均训练集大小（2023）: 无数据\n")

        if test_sizes:
            f.write(f"平均测试集大小（2024）: {np.mean(test_sizes):.1f}\n\n")
        else:
            f.write("平均测试集大小（2024）: 无数据\n\n")

        if len(all_r2) > 0:
            f.write(f"R² 统计:\n")
            f.write(f"  平均值: {np.mean(all_r2):.4f}\n")
            f.write(f"  中位数: {np.median(all_r2):.4f}\n")
            f.write(f"  标准差: {np.std(all_r2):.4f}\n")
            f.write(f"  最大值: {np.max(all_r2):.4f}\n")
            f.write(f"  最小值: {np.min(all_r2):.4f}\n\n")
        else:
            f.write("R² 统计: 无有效模型数据\n\n")

        r2_ranges = {
            'R² > 0.7': sum(1 for r in all_r2 if r > 0.7),
            'R² 0.5-0.7': sum(1 for r in all_r2 if 0.5 <= r <= 0.7),
            'R² 0.3-0.5': sum(1 for r in all_r2 if 0.3 <= r < 0.5),
            'R² < 0.3': sum(1 for r in all_r2 if r < 0.3)
        }

        f.write(f"R² 分布:\n")
        for range_name, count in r2_ranges.items():
            pct = count / len(all_r2) * 100 if all_r2 else 0
            f.write(f"  {range_name}: {count} ({pct:.1f}%)\n")
        f.write("\n")

        f.write(f"RMSE 平均: {np.mean(all_rmse):.4f}\n")
        f.write(f"MAE 平均: {np.mean(all_mae):.4f}\n\n")

        # 参数分布
        param_counts = {}
        for p in all_params:
            param_counts[p] = param_counts.get(p, 0) + 1

        f.write(f"ARIMA参数分布（Top 5）:\n")
        top_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for param, count in top_params:
            f.write(f"  ARIMA{param}: {count} ({count/len(all_params)*100:.1f}%)\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("ARIMA时间序列预测 - 2023训练/2024验证 (Station级别)")
    print("="*80)

    print("\n[1/4] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
        except:
            df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ 总记录数: {len(df)}")

    df['Date'] = pd.to_datetime(df['Date'])
    count_2023 = (df['Date'].dt.year == 2023).sum()
    count_2024 = (df['Date'].dt.year == 2024).sum()
    print(f"✓ 2023年记录数: {count_2023}")
    print(f"✓ 2024年记录数: {count_2024}")

    print("\n[2/4] 建立ARIMA模型...")
    all_results = process_all_stations(df)

    if not all_results:
        print("\n⚠️ 没有符合条件的监测站")
        return

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
    print(f"✓ 结果保存在: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
