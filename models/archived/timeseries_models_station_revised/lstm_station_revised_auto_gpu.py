"""
LSTM时间序列预测 - 仅2023数据 + 自动GPU加速版
LSTM Time Series Prediction - 2023 Only + Auto GPU Detection

⭐ 关键特性：
1. 自动检测并使用CUDA GPU（如果可用）
2. 仅使用2023年的数据
3. 按时间顺序80/20分割训练集和测试集
4. 可视化输出改为折线图（实际值vs预测值随时间变化）
5. 支持CPU/CUDA/MPS(Apple Silicon)

方法说明：
- 使用3层LSTM神经网络（128 hidden units）
- 采用滑动窗口（5个时间步）构建序列数据
- 针对每个监测站的每个指标独立训练模型
- 数据归一化 + 早停策略 + 学习率调度 + 梯度裁剪

适用机型：
- ✅ NVIDIA GPU (CUDA) - 最快
- ✅ Apple Silicon (MPS) - 加速
- ✅ CPU - 通用

依赖库：
pip install torch pandas numpy scikit-learn matplotlib

日期：2025-10-28
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ⭐ 自动检测设备（CUDA GPU / Apple MPS / CPU）
def get_device():
    """
    自动检测并返回最佳可用设备
    优先级: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ 使用CUDA GPU: {device_name}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ 使用Apple Silicon MPS加速")
    else:
        device = torch.device('cpu')
        print(f"✓ 使用CPU")

    return device

# 全局设备设置
DEVICE = get_device()

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录配置
OUTPUT_DIR = 'Outputs/LSTM_2023Only_Station_GPU'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Station_Plots", exist_ok=True)

# 全局配置参数
SEQUENCE_LENGTH = 5   # ⭐ 降低时间序列窗口长度（原30，对2024年数据太长）
MIN_SAMPLES = 40      # 最小样本数要求


def calculate_metrics(y_true, y_pred):
    """计算回归评估指标"""
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


class LSTMModel(nn.Module):
    """
    LSTM神经网络模型

    Architecture:
        - 3层LSTM (128 hidden units, dropout=0.3)
        - 2层全连接网络 (128 -> 64 -> 1)
        - ReLU激活函数 + Dropout正则化
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # 取最后一个时间步
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TimeSeriesDataset(Dataset):
    """PyTorch时间序列数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length):
    """创建时间序列窗口数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


def train_lstm_model(X_train, y_train, X_val, y_val, device, epochs=200, batch_size=16, patience=20):
    """
    训练LSTM模型（支持GPU加速）

    Parameters:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        device: 计算设备（CPU/CUDA/MPS）
        epochs: 最大训练轮数
        batch_size: 批次大小
        patience: 早停耐心值
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ⭐ 将模型移到设备上
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            # ⭐ 将数据移到设备上
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # ⭐ 将数据移到设备上
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"  早停于 Epoch {epoch+1}")  # 减少输出
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history


def predict_indicator(station_data, indicator, device, train_ratio=0.8):
    """
    ⭐ 修改：为单个指标训练LSTM模型（仅使用2023年数据，80/20分割）
    """
    station_data = station_data.copy()
    station_data['Date'] = pd.to_datetime(station_data['Date'])
    station_data = station_data.sort_values('Date').reset_index(drop=True)

    # ⭐ 只使用2023年的数据
    station_data['Year'] = station_data['Date'].dt.year
    data_2023 = station_data[station_data['Year'] == 2023].copy()

    # 检查指标的有效数据量
    indicator_data = data_2023[indicator].dropna()

    # 至少需要20个有效数据点
    if len(indicator_data) < 20:
        return None

    indicator_values = indicator_data.values
    dates = data_2023[data_2023[indicator].notna()]['Date'].values

    # 归一化
    scaler = MinMaxScaler()
    indicator_scaled = scaler.fit_transform(indicator_values.reshape(-1, 1)).flatten()

    # 创建序列
    X, y = create_sequences(indicator_scaled, SEQUENCE_LENGTH)

    if len(X) < 10:  # 至少10个序列
        return None

    # ⭐ 按时间顺序80/20分割
    split_idx = int(len(X) * train_ratio)
    X_train_all, X_test = X[:split_idx], X[split_idx:]
    y_train_all, y_test = y[:split_idx], y[split_idx:]

    if len(X_train_all) < 8 or len(X_test) < 2:
        return None

    # 从训练集中再分出验证集
    val_split = int(len(X_train_all) * 0.8)
    if val_split < 1:
        val_split = 1
    X_train, X_val = X_train_all[:val_split], X_train_all[val_split:]
    y_train, y_val = y_train_all[:val_split], y_train_all[val_split:]

    if len(X_val) < 1:
        X_val = X_train[-1:]
        y_val = y_train[-1:]

    # 训练模型
    model, history = train_lstm_model(X_train, y_train, X_val, y_val, device, epochs=100, patience=15)

    # 预测测试集
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_pred = model(X_test_tensor).squeeze().cpu().numpy()

    # 反归一化
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # 对应的日期
    # 训练部分：从第SEQUENCE_LENGTH个开始到split_idx+SEQUENCE_LENGTH
    train_end_idx = split_idx + SEQUENCE_LENGTH
    train_dates = dates[SEQUENCE_LENGTH:train_end_idx]
    train_values = indicator_values[SEQUENCE_LENGTH:train_end_idx]

    # 测试部分：从split_idx+SEQUENCE_LENGTH开始
    test_dates = dates[split_idx + SEQUENCE_LENGTH:]

    # 计算指标
    metrics = calculate_metrics(y_test_orig, y_test_pred_orig)
    if metrics is None:
        return None

    return {
        'metrics': metrics,
        'y_train': train_values,  # 训练部分的实际值
        'train_dates': train_dates,
        'y_test': y_test_orig,
        'y_pred': y_test_pred_orig,
        'test_dates': test_dates,
        'history': history,
        'train_size': len(train_values),
        'test_size': len(y_test_orig)
    }


def predict_station(station_data, station_name, device):
    """
    ⭐ 修改：预测所有16个水质指标（仅使用2023年数据）
    """
    station_data['Date'] = pd.to_datetime(station_data['Date'])

    # ⭐ 只保留2023年数据
    station_data = station_data[station_data['Date'].dt.year == 2023].copy()

    if len(station_data) < 20:
        print(f"  ⚠️ {station_name}: 2023年数据不足（<20条），跳过")
        return None

    # 所有16个水质指标
    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    results = {}
    print(f"  训练 {station_name} LSTM模型（仅2023年数据）...")

    for indicator in indicators:
        result = predict_indicator(station_data, indicator, device)
        if result:
            results[indicator] = result

    if not results:
        print(f"  ⚠️ {station_name}: 所有指标均失败")
        print(f"     请检查2023年数据量：每个指标需>=20条有效数据")
        return None

    return results


def process_all_stations(df, device):
    """处理所有监测站"""
    print("\n" + "="*80)
    print("LSTM时间序列预测 - 2023训练/2024验证 + GPU加速")
    print("="*80)

    stations = df.groupby(['Province', 'City', 'Monitoring_Station']).size().reset_index(name='count')
    stations = stations[stations['count'] >= MIN_SAMPLES]

    print(f"\n总监测站数: {len(stations)}")
    print(f"设备: {device}")

    all_results = {}
    successful = 0

    for idx, row in stations.iterrows():
        province, city, station = row['Province'], row['City'], row['Monitoring_Station']
        print(f"\n[{idx+1}/{len(stations)}] {province} - {city} - {station}")

        station_data = df[
            (df['Province'] == province) &
            (df['City'] == city) &
            (df['Monitoring_Station'] == station)
        ].copy()

        results = predict_station(station_data, station, device)

        if results:
            station_key = f"{province}_{city}_{station}"
            all_results[station_key] = {
                'province': province,
                'city': city,
                'station': station,
                'results': results
            }
            successful += 1
            # 计算平均R²
            r2_values = [r['metrics']['R2'] for r in results.values()]
            avg_r2 = np.mean(r2_values)
            print(f"  ✓ {len(results)} 指标成功，平均 R² = {avg_r2:.4f}")

    print(f"\n成功: {successful}/{len(stations)}")
    return all_results


def visualize_station(station_key, station_info, output_dir):
    """可视化单个监测站的预测结果（多指标折线图）"""
    results = station_info['results']

    if not results:
        return

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(20, 5*nrows))

    for idx, (indicator, result) in enumerate(results.items()):
        ax = plt.subplot(nrows, ncols, idx + 1)

        # ⭐ 画出2023-2024完整时间序列
        y_train = result['y_train']
        train_dates = pd.to_datetime(result['train_dates'])
        y_test = result['y_test']
        y_pred = result['y_pred']
        test_dates = pd.to_datetime(result['test_dates'])
        m = result['metrics']

        # 训练数据（实际值） - 2023年前80%
        ax.plot(train_dates, y_train, marker='o', linewidth=1.5, markersize=3,
                color='green', label='Train Data', alpha=0.6)
        # 测试数据（实际值） - 2023年后20%
        ax.plot(test_dates, y_test, marker='o', linewidth=2, markersize=5,
                color='steelblue', label='Actual (Test)', alpha=0.8)
        # 测试预测值 - 2023年后20%
        ax.plot(test_dates, y_pred, marker='s', linewidth=2, markersize=5,
                color='orangered', label='Predicted (LSTM)', alpha=0.8, linestyle='--')

        textstr = f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}\nN_test = {m['Sample_Size']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Date (2023 only, 80/20 split)', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    title = f"{station_info['province']} - {station_info['city']} - {station_info['station']}\n" + \
            f"LSTM Prediction (Train: 2023, Test: 2024)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/Station_Plots/{station_key}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_results, output_dir, device):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LSTM时间序列预测报告 - 2023训练/2024验证 + GPU加速\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"计算设备: {device}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("✓ 训练集：仅使用2023年的数据\n")
        f.write("✓ 测试集：2024年的数据\n")
        f.write("✓ 模型：3层LSTM (128 units) + 全连接层\n")
        f.write("✓ 自动GPU加速（CUDA/MPS/CPU）\n")
        f.write("✓ 可视化：折线图显示实际值vs预测值\n\n")

        # 统计（收集所有指标的指标）
        all_r2, all_rmse, all_mae = [], [], []

        for station_info in all_results.values():
            for indicator, result in station_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"监测站数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n")
        if all_r2:
            f.write(f"R² 平均值: {np.mean(all_r2):.4f}\n")
            f.write(f"R² 中位数: {np.median(all_r2):.4f}\n")
            f.write(f"RMSE 平均值: {np.mean(all_rmse):.4f}\n")
            f.write(f"MAE 平均值: {np.mean(all_mae):.4f}\n\n")
        else:
            f.write("无有效数据\n\n")

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

        f.write("\n" + "="*80 + "\n\n")

        # 详细的站点结果
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
            f.write(f"平均R²: {avg_r2:.4f}\n")
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
        f.write("- 每个监测站的每个指标独立建立LSTM时间序列模型\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- 训练集：2023年数据，测试集：2024年数据\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("LSTM时间序列预测 - 2023训练/2024验证 + GPU加速")
    print("="*80)

    print("\n[1/4] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')

    print(f"✓ 总记录数: {len(df)}")

    df['Date'] = pd.to_datetime(df['Date'])
    count_2023 = (df['Date'].dt.year == 2023).sum()
    count_2024 = (df['Date'].dt.year == 2024).sum()
    print(f"✓ 2023年: {count_2023}, 2024年: {count_2024}")

    print("\n[2/4] 训练LSTM模型...")
    all_results = process_all_stations(df, DEVICE)

    if not all_results:
        print("\n⚠️ 没有符合条件的监测站")
        return

    print("\n[3/4] 生成可视化...")
    for idx, (station_key, station_info) in enumerate(all_results.items(), 1):
        if idx % 10 == 0:
            print(f"  进度: {idx}/{len(all_results)}")
        visualize_station(station_key, station_info, OUTPUT_DIR)

    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR, DEVICE)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n✓ 成功建模: {len(all_results)} 个监测站")
    print(f"✓ 结果保存: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
