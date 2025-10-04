"""
LSTM时间序列预测 - TensorFlow/Keras版

方法：
- 使用TensorFlow/Keras实现的2层LSTM神经网络（32/16 units）
- 10个时间步的序列预测
- Keras高级API，代码简洁
- 数据归一化 + EarlyStopping回调

适用机型：
- Linux (推荐) - TensorFlow在Linux上最稳定
- Windows - 需要确保TensorFlow安装正确
- Mac - 不推荐！可能遇到mutex锁问题导致程序卡死

性能：
- 训练速度：慢（存在线程锁竞争）
- 内存需求：中等
- 稳定性：Mac上较差，Linux/Windows较好

注意：
- Mac用户建议使用pytorch版本
- 该版本主要用于对比实验或TensorFlow生态集成
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import os

# 必须在导入TensorFlow之前设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# 配置TensorFlow使用更少的CPU线程，避免锁竞争
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/LSTM_Predictions_TensorFlow'
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


def create_sequences(data, seq_length):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=[32, 16], dropout=0.2):
    """Build LSTM model - 使用更小的模型避免卡死"""
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def build_timeseries_lstm(station_data, indicator, seq_length=10, train_ratio=0.8):
    """
    使用TensorFlow LSTM进行时间序列预测
    """
    data = station_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    # 按时间排序 - 关键！
    data = data.sort_values('Date').reset_index(drop=True)

    # 只保留目标指标和日期
    data = data[['Date', indicator]].dropna()

    if len(data) < seq_length + 30:
        return None

    # 提取目标值
    values = data[indicator].values.reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    # 创建序列
    X, y = create_sequences(scaled_values, seq_length)

    if len(X) < 30:
        return None

    # 时间顺序分割（前80%训练，后20%测试）
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) < 20 or len(X_test) < 3:
        return None

    try:
        # 构建更小的LSTM模型
        model = build_lstm_model(
            input_shape=(seq_length, 1),
            units=[32, 16],  # 减小模型大小
            dropout=0.2
        )

        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,  # 减少patience
            restore_best_weights=True,
            verbose=0
        )

        # 训练模型 - 减少epochs和batch_size
        history = model.fit(
            X_train, y_train,
            epochs=50,  # 减少epochs
            batch_size=16,  # 减小batch_size
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
            workers=1,
            use_multiprocessing=False
        )

        # 预测
        y_pred_scaled = model.predict(X_test, verbose=0)

        # 反归一化
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_original = scaler.inverse_transform(y_test).flatten()

        metrics = calculate_metrics(y_test_original, y_pred)
        if metrics is None:
            return None

        return {
            'model_name': 'LSTM',
            'y_test': y_test_original,
            'y_pred': y_pred,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metrics': metrics,
            'history': history.history
        }

    except Exception as e:
        print(f"    LSTM训练错误: {str(e)}")
        return None


def predict_station(df, province, city, station, seq_length=10):
    """为单个监测站预测所有指标"""
    station_data = df[
        (df['Province'] == province) &
        (df['City'] == city) &
        (df['Monitoring_Station'] == station)
    ].copy()

    if len(station_data) < seq_length + 30:
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
        if valid_count < seq_length + 20:
            continue

        result = build_timeseries_lstm(station_data, indicator, seq_length)
        if result:
            results[indicator] = result

    return results if results else None


def process_all_stations(df, seq_length=10):
    """处理所有监测站"""
    print("\n" + "="*80)
    print("LSTM时间序列预测 - 按监测站建模 (TensorFlow)")
    print("="*80)

    # 获取所有监测站
    stations_info = df.groupby(['Province', 'City', 'Monitoring_Station']).size().reset_index(name='count')
    stations_info = stations_info[stations_info['count'] >= seq_length + 30]  # 至少有足够样本

    print(f"\n总监测站数: {len(stations_info)}")
    print(f"样本数>={seq_length + 30}的监测站: {len(stations_info)}")

    all_results = {}
    successful_stations = 0

    for idx, row in stations_info.iterrows():
        province = row['Province']
        city = row['City']
        station = row['Monitoring_Station']

        print(f"\n[{idx+1}/{len(stations_info)}] {province} - {city} - {station}")
        print(f"  样本数: {row['count']}")

        results = predict_station(df, province, city, station, seq_length)

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

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#3498db')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        textstr = f"Model: LSTM\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{station_info['province']} - {station_info['city']} - {station_info['station']}\nLSTM Time Series Prediction (TensorFlow)"
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
        f.write("LSTM时间序列预测报告 - 按监测站建模 (TensorFlow)\n")
        f.write("LSTM Time Series Prediction Report - Station-Based Modeling\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个监测站独立建模\n")
        f.write("- 使用TensorFlow/Keras LSTM进行时间序列预测\n")
        f.write("- 模型结构：2层LSTM + Dropout + 全连接层\n")
        f.write("- 序列长度：10个时间步\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n")
        f.write("- 使用MinMaxScaler进行数据归一化\n")
        f.write("- 早停策略防止过拟合\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []

        for station_key, station_info in all_results.items():
            for indicator, result in station_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])

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
                f.write(f"{indicator:<30} LSTM           "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个监测站独立建立LSTM时间序列模型\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- LSTM能够捕捉时间序列中的长期依赖关系\n")
        f.write("- 结果按R²从高到低排序\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("LSTM时间序列预测系统 - 按监测站建模 (TensorFlow)")
    print("LSTM Time Series Prediction - Station-Based Modeling")
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
    print("\n[2/4] 按监测站建立LSTM模型...")
    seq_length = 10  # LSTM序列长度
    all_results = process_all_stations(df, seq_length)

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
