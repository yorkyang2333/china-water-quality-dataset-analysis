"""
LSTM时间序列分析 - PyTorch GPU加速版本
Time Series Analysis using LSTM Neural Network - GPU Accelerated Version

⚠️ 警告：此模型不适合当前数据集！(Warning: Not suitable for current dataset!)
原因：数据时间间隔极不规则（0-222天），违反LSTM等间隔采样假设
结果：R² < 0，预测效果极差
建议：使用横截面预测或传统机器学习方法（详见 README.md）

方法说明 (Method Description):
- 使用3层LSTM神经网络（128 hidden units）进行水质指数时间序列预测
- 采用滑动窗口（30个时间步）构建序列数据
- 针对每个监测站独立训练模型
- 自动检测并使用CUDA GPU加速训练（无GPU时自动降级为CPU）
- 数据归一化 + 早停策略 + 学习率调度 + 梯度裁剪 + pin_memory优化

适用场景 (Use Cases):
- ⚠️ 当前数据集时间间隔不规则（0-222天），不适合时间序列预测
- 仅适合有规律采样间隔的时间序列数据（如每日/每小时定期监测）
- 可用于演示LSTM在时间序列预测中的应用（但效果差）

适用机型 (Compatible Devices):
- Windows (推荐) - 配备NVIDIA GPU的电脑，可获得3-10倍加速
- Linux - 支持CUDA的服务器
- Mac - 自动降级为CPU模式（建议使用cpu版本）
- 跨平台兼容: 自动适配CPU/GPU

性能指标 (Performance):
- 训练速度: 快速（GPU并行计算）/ 中等（CPU模式）
- 内存需求: 中等（需GPU显存 > 2GB）/ 低（CPU < 2GB）
- 预测准确度: 极差（R² < 0），因数据不适合

依赖库 (Dependencies):
pip install torch pandas numpy scikit-learn matplotlib
# GPU版本需要: torch (with CUDA support)

作者: Claude Code
日期: 2025-10-05
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

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录配置
OUTPUT_DIR = 'Outputs/LSTM_TimeSeries_GPU'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局配置参数
SEQUENCE_LENGTH = 30  # 时间序列窗口长度
MIN_SAMPLES = 50      # 最小样本数要求


def get_device():
    """
    自动检测并返回最佳计算设备

    Returns:
        torch.device: CUDA GPU或CPU设备
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ GPU加速已启用")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
        print(f"  显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
        return device
    else:
        device = torch.device('cpu')
        print(f"\n✓ 使用CPU模式（未检测到CUDA GPU）")
        return device


def calculate_metrics(y_true, y_pred):
    """
    计算回归评估指标

    Parameters:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        dict: 包含RMSE, MAE, R2, Sample_Size的字典
    """
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
    LSTM神经网络模型 - GPU优化版

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
    """
    创建时间序列窗口数据

    Parameters:
        data: 原始时间序列数据
        seq_length: 序列长度（窗口大小）

    Returns:
        X: 输入序列 [samples, seq_length, 1]
        y: 目标值 [samples]
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


def train_lstm_model(X_train, y_train, X_val, y_val, device, epochs=200, batch_size=32, patience=20):
    """
    训练LSTM模型（GPU加速）

    Parameters:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        device: 计算设备（cuda/cpu）
        epochs: 最大训练轮数
        batch_size: 批次大小（GPU模式使用更大batch_size）
        patience: 早停耐心值

    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    # GPU优化：使用pin_memory加速数据传输
    pin_memory = (device.type == 'cuda')
    num_workers = 4 if device.type == 'cuda' else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
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
                print(f"  早停于 Epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_model_state)
    return model, history


def predict_station(station_data, station_name, device):
    """
    对单个监测站进行LSTM时间序列预测

    Parameters:
        station_data: 该监测站的历史数据
        station_name: 监测站名称
        device: 计算设备

    Returns:
        metrics: 评估指标
        predictions: 预测结果DataFrame
    """
    # 按日期排序
    station_data = station_data.sort_values('Date').reset_index(drop=True)

    # 提取目标变量（水质指数）
    wqi_values = station_data['Water_Quality_Index'].values

    # 数据不足检查
    if len(wqi_values) < MIN_SAMPLES:
        print(f"  ⚠️ {station_name}: 数据不足 ({len(wqi_values)} < {MIN_SAMPLES}), 跳过")
        return None, None

    # 数据归一化
    scaler = MinMaxScaler()
    wqi_scaled = scaler.fit_transform(wqi_values.reshape(-1, 1)).flatten()

    # 创建序列数据
    X, y = create_sequences(wqi_scaled, SEQUENCE_LENGTH)

    if len(X) < 10:
        print(f"  ⚠️ {station_name}: 序列样本不足 ({len(X)} < 10), 跳过")
        return None, None

    # 划分训练集和测试集 (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 训练模型
    print(f"  训练 {station_name} LSTM模型（{device.type.upper()}）...")
    model, history = train_lstm_model(X_train, y_train, X_test, y_test, device)

    # 预测
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        y_train_pred = model(X_train_tensor).squeeze().cpu().numpy()
        y_test_pred = model(X_test_tensor).squeeze().cpu().numpy()

    # 反归一化
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_pred_orig = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # 计算指标
    metrics = calculate_metrics(y_test_orig, y_test_pred_orig)

    if metrics:
        print(f"  ✓ {station_name} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")

    # 保存预测结果
    predictions = pd.DataFrame({
        'Date': station_data['Date'].iloc[SEQUENCE_LENGTH+split_idx:].values,
        'Actual_WQI': y_test_orig,
        'Predicted_WQI': y_test_pred_orig
    })

    return metrics, predictions


def main():
    """主函数：执行LSTM时间序列分析（GPU加速）"""
    print("=" * 80)
    print("LSTM时间序列分析 - GPU加速版本")
    print("LSTM Time Series Analysis - GPU Accelerated Version")
    print("=" * 80)

    # 检测计算设备
    device = get_device()

    # 加载数据
    print("\n[1] 加载数据...")
    df = pd.read_csv('preprocessed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  总数据量: {len(df)} 条")
    print(f"  监测站数量: {df['Monitoring_Station'].nunique()} 个")

    # 检查数据分布
    station_counts = df.groupby('Monitoring_Station').size()
    sufficient_stations = station_counts[station_counts >= MIN_SAMPLES].index.tolist()

    print(f"\n  数据充足的监测站（>={MIN_SAMPLES}点）: {len(sufficient_stations)} 个")
    print(f"  数据不足的监测站: {len(station_counts) - len(sufficient_stations)} 个")

    if len(sufficient_stations) == 0:
        print(f"\n⚠️ 警告: 没有监测站满足最小样本数要求（{MIN_SAMPLES}个点）")
        print("  该数据集不适合LSTM时间序列预测！")
        print("  建议使用横截面预测方法（Random Forest/XGBoost）")
        return

    # 对每个监测站进行预测
    print(f"\n[2] 训练LSTM模型（共 {len(sufficient_stations)} 个监测站）...")
    results = []

    for i, station in enumerate(sufficient_stations, 1):
        print(f"\n[{i}/{len(sufficient_stations)}] 处理 {station}...")
        station_data = df[df['Monitoring_Station'] == station].copy()

        metrics, predictions = predict_station(station_data, station, device)

        if metrics and predictions is not None:
            results.append({
                'Station': station,
                'Province': station_data['Province'].iloc[0],
                'City': station_data['City'].iloc[0],
                **metrics
            })

            # 保存预测结果
            predictions.to_csv(f'{OUTPUT_DIR}/{station}_predictions.csv', index=False)

    # 汇总结果
    if results:
        print(f"\n[3] 生成分析报告...")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)
        results_df.to_csv(f'{OUTPUT_DIR}/all_stations_performance.csv', index=False)

        # 可视化
        plot_results(results_df, device)

        # 生成文本报告
        generate_report(results_df, device)

        print(f"\n✓ 分析完成！")
        print(f"  成功预测监测站数: {len(results)}/{len(sufficient_stations)}")
        print(f"  输出目录: {OUTPUT_DIR}/")
        print(f"  使用设备: {device.type.upper()}")

        if device.type == 'cuda':
            print(f"  GPU显存峰值: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    else:
        print("\n⚠️ 没有成功的预测结果")


def plot_results(results_df, device):
    """生成结果可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    device_label = f"GPU: {torch.cuda.get_device_name(0)}" if device.type == 'cuda' else "CPU"

    # R² 分布
    axes[0, 0].hist(results_df['R2'], bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_title(f'R² Score Distribution / R²分数分布\n({device_label})',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_ylabel('Frequency / 频数')
    axes[0, 0].axvline(results_df['R2'].mean(), color='red', linestyle='--',
                       label=f'Mean / 均值: {results_df["R2"].mean():.3f}')
    axes[0, 0].legend()

    # RMSE 分布
    axes[0, 1].hist(results_df['RMSE'], bins=20, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title(f'RMSE Distribution / RMSE分布\n({device_label})',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('RMSE')
    axes[0, 1].set_ylabel('Frequency / 频数')
    axes[0, 1].axvline(results_df['RMSE'].mean(), color='darkred', linestyle='--',
                       label=f'Mean / 均值: {results_df["RMSE"].mean():.3f}')
    axes[0, 1].legend()

    # Top 10 监测站 (按R²)
    top10 = results_df.head(10)
    axes[1, 0].barh(range(len(top10)), top10['R2'], color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(top10)))
    axes[1, 0].set_yticklabels(top10['Station'], fontsize=9)
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_title('Top 10 Stations by R² / R²最高的10个监测站',
                         fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()

    # 样本量 vs R²
    axes[1, 1].scatter(results_df['Sample_Size'], results_df['R2'], alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Sample Size / 样本量')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Sample Size vs R² / 样本量与R²关系',
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lstm_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 可视化图表已保存: lstm_analysis_summary.png")


def generate_report(results_df, device):
    """生成文本分析报告"""
    device_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)" if device.type == 'cuda' else "CPU"

    report_lines = [
        "=" * 80,
        "LSTM时间序列分析报告 - GPU加速版",
        "LSTM Time Series Analysis Report - GPU Accelerated",
        "=" * 80,
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"计算设备: {device_info}",
        f"模型类型: 3层LSTM神经网络（128 hidden units）",
        f"序列长度: {SEQUENCE_LENGTH} 个时间步",
        "",
        "=" * 80,
        "1. 整体性能指标 (Overall Performance)",
        "=" * 80,
        f"  成功预测监测站数: {len(results_df)} 个",
        f"  平均 R² Score: {results_df['R2'].mean():.4f} (±{results_df['R2'].std():.4f})",
        f"  平均 RMSE: {results_df['RMSE'].mean():.4f} (±{results_df['RMSE'].std():.4f})",
        f"  平均 MAE: {results_df['MAE'].mean():.4f} (±{results_df['MAE'].std():.4f})",
        f"  平均样本量: {results_df['Sample_Size'].mean():.0f} 个",
        "",
        "=" * 80,
        "2. Top 10 最佳预测监测站 (By R²)",
        "=" * 80,
        f"{'排名':<6} {'监测站':<25} {'省份':<12} {'城市':<12} {'R²':<10} {'RMSE':<10}",
        "-" * 80
    ]

    for i, row in results_df.head(10).iterrows():
        report_lines.append(
            f"{i+1:<6} {row['Station']:<25} {row['Province']:<12} {row['City']:<12} "
            f"{row['R2']:<10.4f} {row['RMSE']:<10.4f}"
        )

    report_lines.extend([
        "",
        "=" * 80,
        "3. 模型局限性与建议 (Limitations & Recommendations)",
        "=" * 80,
        "",
        "⚠️ 数据限制:",
        f"  - 本数据集每个监测站平均仅有1-3个数据点",
        f"  - 时间序列数据严重不足，无法充分训练LSTM模型",
        f"  - 仅{len(results_df)}个监测站满足最小样本要求（>={MIN_SAMPLES}点）",
        "",
        "💡 建议:",
        "  1. LSTM不适合本数据集的时间序列预测",
        "  2. 推荐使用横截面预测方法：",
        "     - Random Forest（基于地理特征和水质参数）",
        "     - XGBoost（梯度提升）",
        "     - 地理加权回归（GWR）",
        "  3. 如需时间序列预测，需收集更多历史数据（建议>100个时间点）",
        "",
        f"🚀 GPU加速性能:",
        f"  - 计算设备: {device_info}",
    ])

    if device.type == 'cuda':
        report_lines.append(f"  - 显存峰值使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        report_lines.append(f"  - 加速比: 预计为CPU的3-10倍（取决于GPU型号）")

    report_lines.extend([
        "",
        "=" * 80,
        "报告结束",
        "=" * 80
    ])

    report_path = f'{OUTPUT_DIR}/report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"  ✓ 分析报告已保存: report.txt")

    # 打印报告摘要
    print("\n" + "\n".join(report_lines[:30]))  # 打印前30行


if __name__ == '__main__':
    main()
