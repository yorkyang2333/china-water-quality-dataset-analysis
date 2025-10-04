import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/TimeSeries_Station_Predictions'
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


def build_timeseries_model(station_data, indicator, train_ratio=0.8):
    """
    纯时间序列分析：只使用历史值、时间特征
    不使用其他水质指标作为特征
    """
    data = station_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    # 按时间排序 - 关键！
    data = data.sort_values('Date').reset_index(drop=True)

    # 只保留目标指标和日期
    data = data[['Date', indicator]].dropna()

    if len(data) < 30:
        return None

    # 提取时间特征
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Quarter'] = data['Date'].dt.quarter

    # 添加滞后特征（历史值）
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        data[f'lag_{lag}'] = data[indicator].shift(lag)

    # 添加滚动统计特征
    for window in [3, 7, 14, 30]:
        data[f'rolling_mean_{window}'] = data[indicator].rolling(window=window, min_periods=1).mean()
        data[f'rolling_std_{window}'] = data[indicator].rolling(window=window, min_periods=1).std().fillna(0)
        data[f'rolling_min_{window}'] = data[indicator].rolling(window=window, min_periods=1).min()
        data[f'rolling_max_{window}'] = data[indicator].rolling(window=window, min_periods=1).max()

    # 添加差分特征
    data['diff_1'] = data[indicator].diff(1).fillna(0)
    data['diff_7'] = data[indicator].diff(7).fillna(0)

    # 特征列表
    time_features = ['Year', 'Month', 'DayOfYear', 'DayOfWeek', 'Quarter']
    lag_features = [col for col in data.columns if col.startswith('lag_')]
    rolling_features = [col for col in data.columns if col.startswith('rolling_')]
    diff_features = ['diff_1', 'diff_7']

    feature_cols = time_features + lag_features + rolling_features + diff_features

    # 删除NaN行
    model_data = data.dropna()

    if len(model_data) < 20:
        return None

    # 时间顺序分割（前80%训练，后20%测试）
    split_idx = int(len(model_data) * train_ratio)
    train_data = model_data.iloc[:split_idx]
    test_data = model_data.iloc[split_idx:]

    if len(train_data) < 15 or len(test_data) < 3:
        return None

    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values
    X_test = test_data[feature_cols].values
    y_test = test_data[indicator].values

    try:
        # 尝试两个模型
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge': Ridge(alpha=1.0)
        }

        best_r2 = -np.inf
        best_model_name = None
        best_pred = None

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_pred = y_pred

        metrics = calculate_metrics(y_test, best_pred)
        if metrics is None:
            return None

        return {
            'model_name': best_model_name,
            'y_test': y_test,
            'y_pred': best_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics
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
        # 检查数据量
        valid_count = station_data[indicator].notna().sum()
        if valid_count < 20:
            continue

        result = build_timeseries_model(station_data, indicator)
        if result:
            results[indicator] = result

    return results if results else None


def process_all_stations(df):
    """处理所有监测站"""
    print("\n" + "="*80)
    print("时间序列预测 - 按监测站建模")
    print("="*80)

    # 获取所有监测站
    stations_info = df.groupby(['Province', 'City', 'Monitoring_Station']).size().reset_index(name='count')
    stations_info = stations_info[stations_info['count'] >= 30]  # 至少30个样本

    print(f"\n总监测站数: {len(stations_info)}")
    print(f"样本数>=30的监测站: {len(stations_info)}")

    all_results = {}
    successful_stations = 0

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

            # 显示结果摘要
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

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#2ecc71')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        textstr = f"Model: {result['model_name']}\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{station_info['province']} - {station_info['city']} - {station_info['station']}\nTime Series Prediction"
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
        f.write("时间序列预测报告 - 按监测站建模\n")
        f.write("Time Series Prediction Report - Station-Based Modeling\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个监测站独立建模（100个监测站）\n")
        f.write("- 纯时间序列分析：只使用历史值和时间特征\n")
        f.write("- 特征包括：滞后值、滚动统计、差分、时间特征\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n")
        f.write("- 模型：RandomForest vs Ridge，自动选择最佳\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []
        model_counts = {}

        for station_key, station_info in all_results.items():
            for indicator, result in station_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                model_counts[result['model_name']] = model_counts.get(result['model_name'], 0) + 1

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"监测站数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n\n")

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

        f.write("模型使用统计:\n")
        for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(all_r2) * 100
            f.write(f"  {model_name}: {count} ({percentage:.1f}%)\n")

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

        # 收集所有站点的平均R²
        station_avg_r2 = []
        for station_key, station_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in station_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            station_avg_r2.append((station_key, station_info, avg_r2))

        # 按平均R²排序
        station_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for station_key, station_info, avg_r2 in station_avg_r2:
            f.write(f"监测站: {station_info['province']} - {station_info['city']} - {station_info['station']}\n")
            f.write(f"样本数: {station_info['sample_count']} | 平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'模型':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            # 按R²排序
            sorted_results = sorted(station_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                f.write(f"{indicator:<30} {result['model_name']:<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个监测站独立建立时间序列模型\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- 使用滞后特征、滚动统计、时间特征进行预测\n")
        f.write("- 结果按R²从高到低排序\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("时间序列预测系统 - 按监测站建模")
    print("Time Series Prediction - Station-Based Modeling")
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

    # 处理所有监测站
    print("\n[2/4] 按监测站建模...")
    all_results = process_all_stations(df)

    # 生成可视化
    print("\n[3/4] 生成可视化...")
    total = len(all_results)
    for idx, (station_key, station_info) in enumerate(all_results.items(), 1):
        if idx % 10 == 0 or idx == total:
            print(f"  进度: {idx}/{total}")
        visualize_station(station_key, station_info, OUTPUT_DIR)

    # 生成报告
    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR)

    # 总结
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
