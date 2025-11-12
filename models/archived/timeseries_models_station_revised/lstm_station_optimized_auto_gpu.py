"""
LSTM时间序列预测 - ⭐⭐ 优化版 + Optuna自动调参 + 模型保存
LSTM Time Series Prediction - Optimized with Auto Hyperparameter Tuning

⭐⭐ 新增特性（解决直线预测问题）：
1. ✅ Optuna自动调参（学习率、隐藏层、SEQUENCE_LENGTH等）
2. ✅ 优化早停策略（patience=50，min_delta=1e-5）
3. ✅ 保存最佳模型为.pth文件
4. ✅ 可调节SEQUENCE_LENGTH（10-30范围自动搜索）
5. ✅ 改进的学习率调度（CosineAnnealingLR）
6. ✅ LayerNorm提高训练稳定性
7. ✅ 更长的训练时间（最多300 epochs）

⭐ 原有特性：
- 自动检测并使用CUDA GPU/MPS/CPU
- 仅使用2023年数据，80/20分割
- 预测所有16个水质指标
- 完整的可视化和报告

问题诊断：
- ❌ 旧版SEQUENCE_LENGTH=5太短 → ✅ 现在10-30自动搜索
- ❌ 旧版patience=15太小 → ✅ 现在patience=50
- ❌ 旧版缺少调参 → ✅ 现在Optuna自动调参
- ❌ 旧版epochs=100不够 → ✅ 现在最多300 epochs

依赖库：
pip install torch pandas numpy scikit-learn matplotlib optuna

作者：Claude Code
日期：2025-10-29
版本：v2.0 (Optimized)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ⭐⭐ NEW: Optuna for hyperparameter tuning
import optuna
from optuna.trial import TrialState

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ⭐ 自动检测设备
def get_device():
    """自动检测并返回最佳可用设备"""
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

DEVICE = get_device()

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = 'Outputs/LSTM_2023Only_Station_GPU_Optimized'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Station_Plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Saved_Models", exist_ok=True)

# ⭐⭐ 优化后的全局参数
MIN_SAMPLES = 40


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


class ImprovedLSTMModel(nn.Module):
    """
    ⭐⭐ 改进的LSTM模型

    新特性：
    - LayerNorm for stability
    - 可调节的hidden_size和num_layers
    - 更好的dropout策略
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # ⭐ 添加LayerNorm提高稳定性
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步
        out = self.layer_norm(out)  # ⭐ LayerNorm
        out = self.dropout(out)
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


def train_lstm_model(X_train, y_train, X_val, y_val, device,
                     hidden_size=128, num_layers=2, dropout=0.2,
                     lr=0.001, epochs=300, batch_size=16, patience=50):
    """
    ⭐⭐ 优化后的训练函数

    改进：
    - patience=50（原15） → 避免过早停止
    - epochs=300（原100） → 更充分的训练
    - min_delta=1e-5 → 更敏感的早停检测
    - CosineAnnealingLR → 更好的学习率调度
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ImprovedLSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ⭐ 改进的学习率调度：CosineAnnealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    min_delta = 1e-5  # ⭐ 最小改进阈值

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        scheduler.step()

        # ⭐⭐ 改进的早停检查（使用min_delta）
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def objective(trial, X_train, y_train, X_val, y_val, device):
    """
    ⭐⭐ Optuna目标函数：自动搜索最佳超参数

    搜索空间：
    - hidden_size: 64-256
    - num_layers: 2-4
    - dropout: 0.1-0.4
    - lr: 1e-4 to 1e-2
    - batch_size: 8-32
    """
    hidden_size = trial.suggest_int('hidden_size', 64, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)

    model, history = train_lstm_model(
        X_train, y_train, X_val, y_val, device,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=150,  # 调参时用较少epochs
        batch_size=batch_size,
        patience=30
    )

    # 返回验证集最佳loss
    return min(history['val_loss'])


def predict_indicator_with_tuning(station_data, indicator, device, station_name, train_ratio=0.8):
    """
    ⭐⭐ 带Optuna自动调参的预测函数

    改进：
    - 自动搜索最佳SEQUENCE_LENGTH（10-30）
    - 自动搜索最佳模型参数
    - 保存最佳模型为.pth
    """
    station_data = station_data.copy()
    station_data['Date'] = pd.to_datetime(station_data['Date'])
    station_data = station_data.sort_values('Date').reset_index(drop=True)

    # 只使用2023年数据
    station_data['Year'] = station_data['Date'].dt.year
    data_2023 = station_data[station_data['Year'] == 2023].copy()

    indicator_data = data_2023[indicator].dropna()
    if len(indicator_data) < 40:
        return None

    indicator_values = indicator_data.values
    dates = data_2023[data_2023[indicator].notna()]['Date'].values

    # 归一化
    scaler = MinMaxScaler()
    indicator_scaled = scaler.fit_transform(indicator_values.reshape(-1, 1)).flatten()

    # ⭐⭐ Step 1: 搜索最佳SEQUENCE_LENGTH
    best_seq_length = None
    best_seq_score = float('inf')

    print(f"      [1/3] 搜索最佳SEQUENCE_LENGTH...", end=" ")
    for seq_len in [10, 15, 20, 25, 30]:
        X, y = create_sequences(indicator_scaled, seq_len)
        if len(X) < 20:
            continue

        split_idx = int(len(X) * train_ratio)
        if split_idx < 10:
            continue

        X_train_all, X_test = X[:split_idx], X[split_idx:]
        y_train_all, y_test = y[:split_idx], y[split_idx:]

        val_split = int(len(X_train_all) * 0.8)
        X_train, X_val = X_train_all[:val_split], X_train_all[val_split:]
        y_train, y_val = y_train_all[:val_split], y_train_all[val_split:]

        if len(X_val) < 2:
            continue

        # 快速测试
        model, history = train_lstm_model(
            X_train, y_train, X_val, y_val, device,
            hidden_size=128, num_layers=2, dropout=0.2,
            lr=0.001, epochs=50, batch_size=16, patience=15
        )

        val_loss = min(history['val_loss'])
        if val_loss < best_seq_score:
            best_seq_score = val_loss
            best_seq_length = seq_len

    if best_seq_length is None:
        print("失败")
        return None

    print(f"最佳={best_seq_length}")

    # 使用最佳SEQUENCE_LENGTH创建最终数据
    X, y = create_sequences(indicator_scaled, best_seq_length)
    split_idx = int(len(X) * train_ratio)
    X_train_all, X_test = X[:split_idx], X[split_idx:]
    y_train_all, y_test = y[:split_idx], y[split_idx:]

    val_split = int(len(X_train_all) * 0.8)
    X_train, X_val = X_train_all[:val_split], X_train_all[val_split:]
    y_train, y_val = y_train_all[:val_split], y_train_all[val_split:]

    # ⭐⭐ Step 2: Optuna自动调参
    print(f"      [2/3] Optuna自动调参 (20 trials)...")
    study = optuna.create_study(direction='minimize', study_name=f'{station_name}_{indicator}')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, device),
        n_trials=20,
        show_progress_bar=False
    )

    best_params = study.best_params
    print(f"      最佳参数: hidden={best_params['hidden_size']}, layers={best_params['num_layers']}, "
          f"lr={best_params['lr']:.5f}, dropout={best_params['dropout']:.2f}")

    # ⭐⭐ Step 3: 使用最佳参数训练最终模型
    print(f"      [3/3] 训练最终模型...")
    final_model, history = train_lstm_model(
        X_train_all, y_train_all, X_val, y_val, device,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        lr=best_params['lr'],
        epochs=300,
        batch_size=best_params['batch_size'],
        patience=50
    )

    # ⭐⭐ 保存最佳模型
    model_filename = f"{OUTPUT_DIR}/Saved_Models/{station_name}_{indicator}.pth"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'best_params': best_params,
        'seq_length': best_seq_length,
        'scaler_min': scaler.data_min_[0],
        'scaler_max': scaler.data_max_[0]
    }, model_filename)

    # 预测测试集
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_pred = final_model(X_test_tensor).squeeze().cpu().numpy()

    # 反归一化
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # 对应的日期
    train_end_idx = split_idx + best_seq_length
    train_dates = dates[best_seq_length:train_end_idx]
    train_values = indicator_values[best_seq_length:train_end_idx]
    test_dates = dates[split_idx + best_seq_length:]

    # 计算指标
    metrics = calculate_metrics(y_test_orig, y_test_pred_orig)
    if metrics is None:
        return None

    return {
        'metrics': metrics,
        'y_train': train_values,
        'train_dates': train_dates,
        'y_test': y_test_orig,
        'y_pred': y_test_pred_orig,
        'test_dates': test_dates,
        'history': history,
        'train_size': len(train_values),
        'test_size': len(y_test_orig),
        'best_params': best_params,
        'seq_length': best_seq_length,
        'model_file': model_filename
    }


def predict_station(station_data, station_name, device):
    """预测所有16个水质指标"""
    station_data['Date'] = pd.to_datetime(station_data['Date'])
    station_data = station_data[station_data['Date'].dt.year == 2023].copy()

    if len(station_data) < 40:
        print(f"  ⚠️ {station_name}: 2023年数据不足（<40条），跳过")
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
    print(f"  训练 {station_name} LSTM模型（⭐⭐ Optuna自动调参）...")

    for idx, indicator in enumerate(indicators, 1):
        print(f"    [{idx}/{len(indicators)}] {indicator}:")
        result = predict_indicator_with_tuning(station_data, indicator, device, station_name)
        if result:
            results[indicator] = result
            print(f"      ✓ R²={result['metrics']['R2']:.4f}, 模型已保存: {result['model_file']}")

    if not results:
        print(f"  ⚠️ {station_name}: 所有指标均失败")
        return None

    return results


def process_all_stations(df, device, max_stations=None):
    """处理所有监测站（可限制数量用于测试）"""
    print("\n" + "="*80)
    print("LSTM时间序列预测 - ⭐⭐ Optuna自动调参版 (Station级别)")
    print("="*80)

    df['Station_Key'] = df['Province'] + '_' + df['City'] + '_' + df['Station']
    stations = df['Station_Key'].unique()

    if max_stations:
        stations = stations[:max_stations]
        print(f"\n⚠️ 测试模式：仅处理前{max_stations}个监测站")

    print(f"\n总监测站数: {len(stations)}")
    print(f"设备: {device}")

    all_results = {}
    successful = 0

    for idx, station_key in enumerate(stations, 1):
        print(f"\n[{idx}/{len(stations)}] {station_key}")

        station_data = df[df['Station_Key'] == station_key].copy()
        results = predict_station(station_data, station_key, device)

        if results:
            province, city, station = station_key.split('_', 2)
            all_results[station_key] = {
                'province': province,
                'city': city,
                'station': station,
                'results': results
            }
            successful += 1
            r2_values = [r['metrics']['R2'] for r in results.values()]
            avg_r2 = np.mean(r2_values)
            print(f"  ✓ {len(results)} 指标成功，平均 R² = {avg_r2:.4f}")

    print(f"\n成功: {successful}/{len(stations)}")
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

        y_train = result['y_train']
        train_dates = pd.to_datetime(result['train_dates'])
        y_test = result['y_test']
        y_pred = result['y_pred']
        test_dates = pd.to_datetime(result['test_dates'])
        m = result['metrics']

        # 训练数据
        ax.plot(train_dates, y_train, marker='o', linewidth=1.5, markersize=3,
                color='green', label='Train Data', alpha=0.6)
        # 测试数据（实际值）
        ax.plot(test_dates, y_test, marker='o', linewidth=2, markersize=5,
                color='steelblue', label='Actual (Test)', alpha=0.8)
        # 预测值
        ax.plot(test_dates, y_pred, marker='s', linewidth=2, markersize=5,
                color='orangered', label='Predicted (LSTM)', alpha=0.8, linestyle='--')

        # ⭐ 显示调参信息
        params = result['best_params']
        textstr = (f"R² = {m['R2']:.3f}\n"
                   f"RMSE = {m['RMSE']:.2f}\n"
                   f"MAE = {m['MAE']:.2f}\n"
                   f"SeqLen = {result['seq_length']}\n"
                   f"Hidden = {params['hidden_size']}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Date (2023 only, 80/20 split)', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    title = f"{station_info['station']} Station ({station_info['city']}, {station_info['province']})\n" + \
            f"LSTM Prediction (⭐⭐ Optimized + Auto-Tuned)"
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
        f.write("LSTM时间序列预测报告 - ⭐⭐ Optuna自动调参优化版\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"计算设备: {device}\n\n")

        f.write("⭐⭐ 优化特性：\n")
        f.write("-" * 80 + "\n")
        f.write("✅ Optuna自动调参（hidden_size, num_layers, lr, dropout, batch_size）\n")
        f.write("✅ 自动搜索最佳SEQUENCE_LENGTH（10-30）\n")
        f.write("✅ 优化早停策略（patience=50, min_delta=1e-5）\n")
        f.write("✅ 最佳模型保存为.pth文件\n")
        f.write("✅ CosineAnnealingLR学习率调度\n")
        f.write("✅ LayerNorm提高训练稳定性\n")
        f.write("✅ 最多300 epochs充分训练\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("✓ ⭐ 仅使用2023年的数据\n")
        f.write("✓ ⭐ 训练集：2023年数据的前80%（按时间顺序）\n")
        f.write("✓ ⭐ 测试集：2023年数据的后20%（按时间顺序）\n")
        f.write("✓ 模型：改进的LSTM + LayerNorm\n")
        f.write("✓ 自动GPU加速（CUDA/MPS/CPU）\n\n")

        # 统计
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

        f.write("详细结果 - 按R²排序\n")
        f.write("="*80 + "\n\n")

        # 详细结果
        station_avg_r2 = []
        for station_key, station_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in station_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            station_avg_r2.append((station_key, station_info, avg_r2))

        station_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for station_key, station_info, avg_r2 in station_avg_r2:
            f.write(f"监测站: {station_info['station']} ({station_info['city']}, {station_info['province']})\n")
            f.write(f"平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'SeqLen':>7} {'Hidden':>7} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            sorted_results = sorted(station_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                params = result['best_params']
                f.write(f"{indicator:<30} "
                       f"{result['seq_length']:>7} "
                       f"{params['hidden_size']:>7} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个监测站的每个指标独立使用Optuna自动调参\n")
        f.write("- SeqLen: 自动搜索的最佳SEQUENCE_LENGTH\n")
        f.write("- Hidden: 自动搜索的最佳隐藏层大小\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- ⭐ 仅使用2023年数据，按时间顺序80/20分割\n")
        f.write("- ⭐⭐ 所有最佳模型已保存至 Saved_Models/ 文件夹\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("LSTM时间序列预测 - ⭐⭐ Optuna自动调参优化版 (Station级别)")
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
    print(f"⭐ 本次分析仅使用2023年数据")

    # ⚠️ 测试模式：只处理前5个站点（正式运行时删除这行）
    print("\n[2/4] 训练LSTM模型...")
    all_results = process_all_stations(df, DEVICE, max_stations=5)

    if not all_results:
        print("\n⚠️ 没有符合条件的监测站")
        return

    print("\n[3/4] 生成可视化...")
    for idx, (station_key, station_info) in enumerate(all_results.items(), 1):
        print(f"  进度: {idx}/{len(all_results)}")
        visualize_station(station_key, station_info, OUTPUT_DIR)

    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR, DEVICE)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n✓ 成功建模: {len(all_results)} 个监测站")
    print(f"✓ 结果保存: {OUTPUT_DIR}/")
    print(f"✓ 模型保存: {OUTPUT_DIR}/Saved_Models/\n")


if __name__ == "__main__":
    main()
