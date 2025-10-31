"""
🔍 北京省LSTM调试脚本
Debug Script for Beijing Province Only

用途：
- 仅对北京省的水质因子进行预测
- 快速验证修复效果
- 便于调试和参数调整

特点：
- 只训练2个指标：pH 和 Water_Temperature_C
- 更详细的输出日志
- 可视化训练过程
- 显示数据聚合前后对比

使用方法：
python debug_beijing_only.py

作者：Claude Code
日期：2025-10-31
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

import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def get_device():
    """自动检测设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ 使用Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print(f"✓ 使用CPU")
    return device


DEVICE = get_device()

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/DEBUG_Beijing_Only'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
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


def visualize_data_aggregation(df_raw, df_daily, indicator, output_dir):
    """
    🔍 可视化：数据聚合前后对比

    显示原始数据（多个监测站）vs 按日聚合后的数据
    """
    plt.figure(figsize=(14, 6))

    # 原始数据（多个监测站）
    plt.subplot(1, 2, 1)
    stations = df_raw['Monitoring_Station'].unique()[:5]  # 只显示前5个站点
    for station in stations:
        station_data = df_raw[df_raw['Monitoring_Station'] == station]
        plt.plot(station_data['Date'], station_data[indicator],
                marker='o', markersize=3, alpha=0.6, label=station[:15])

    plt.title('原始数据（多个监测站，未聚合）', fontsize=11, fontweight='bold')
    plt.xlabel('Date', fontsize=9)
    plt.ylabel(indicator, fontsize=9)
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 按日聚合后的数据
    plt.subplot(1, 2, 2)
    plt.plot(df_daily.index, df_daily.values,
            marker='o', markersize=4, linewidth=2, color='darkblue', alpha=0.8)
    plt.title('按日聚合后（省级平均值）', fontsize=11, fontweight='bold')
    plt.xlabel('Date', fontsize=9)
    plt.ylabel(indicator, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.suptitle(f'北京 - {indicator}\n数据聚合前后对比',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = f'{output_dir}/aggregation_comparison_{indicator}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 聚合对比图已保存: {filename}")


def build_daily_series_debug(province_df, indicator, output_dir):
    """
    🔍 按日聚合（调试版）

    额外功能：
    - 显示数据量变化
    - 可视化聚合前后对比
    - 显示缺失日期填充情况
    """
    df = province_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2023]

    print(f"\n  🔍 数据聚合详情（{indicator}）：")
    print(f"    原始数据点数: {len(df)}")
    print(f"    监测站数量: {df['Monitoring_Station'].nunique()}")

    # 按日聚合
    daily = (df.groupby('Date')[indicator]
               .mean()
               .sort_index())

    print(f"    聚合后数据点数: {len(daily)}")

    # 🔍 可视化聚合前后对比
    visualize_data_aggregation(df, daily, indicator, output_dir)

    # 连续日期 + 插值
    daily_continuous = (daily
                       .asfreq('D')
                       .interpolate(method='linear', limit_direction='both'))

    filled_count = daily_continuous.isna().sum()
    if filled_count > 0:
        print(f"    ⚠️ 填充缺失日期: {filled_count} 天")
    print(f"    最终连续序列长度: {len(daily_continuous)}")

    # 检查数据量
    if len(daily_continuous) < 60:
        print(f"    ❌ 数据不足（<60天）")
        return None

    # 时间切分
    split_idx = int(len(daily_continuous) * 0.8)
    y_train = daily_continuous.iloc[:split_idx].values
    y_test = daily_continuous.iloc[split_idx:].values

    print(f"    训练集: {len(y_train)} 天")
    print(f"    测试集: {len(y_test)} 天")

    # 只用训练段拟合缩放器
    scaler = MinMaxScaler()
    y_train_sc = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_sc = scaler.transform(y_test.reshape(-1, 1)).ravel()

    return daily_continuous.index, y_train_sc, y_test_sc, y_train, y_test, scaler


class FixedLSTMModel(nn.Module):
    """修复版LSTM模型"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.15):
        super(FixedLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.norm(out)
        return self.head(out)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length):
    """创建序列"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


def calculate_naive_baseline(y_train, y_val):
    """Naive baseline"""
    y_val_pred = np.roll(y_val, 1)
    y_val_pred[0] = y_train[-1] if len(y_train) > 0 else y_val[0]
    return mean_squared_error(y_val, y_val_pred)


def train_lstm_model(X_train, y_train, X_val, y_val, device,
                     hidden_size=128, num_layers=2, dropout=0.15,
                     lr=0.001, epochs=60, batch_size=32, patience=12,
                     verbose=True):
    """训练模型（带详细输出）"""
    baseline_mse = calculate_naive_baseline(y_train, y_val)

    if verbose:
        print(f"      Naive baseline MSE: {baseline_mse:.6f}")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = FixedLSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
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

        scheduler.step()

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

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"      早停于 Epoch {epoch+1}/{epochs}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1:3d}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

    if best_val_loss > baseline_mse * 1.1:
        print(f"      ⚠️ 模型未超过baseline!")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, baseline_mse


def objective(trial, y_train_sc, y_test_sc, device):
    """Optuna目标函数"""
    seq_len = trial.suggest_categorical('seq_len', [5, 10, 15, 20, 25, 30])
    X, y = create_sequences(y_train_sc, seq_len)

    if len(X) < 30:
        raise optuna.TrialPruned()

    val_split = int(len(X) * 0.8)
    if val_split < 15:
        raise optuna.TrialPruned()

    X_tr, y_tr = X[:val_split], y[:val_split]
    X_va, y_va = X[val_split:], y[val_split:]

    if len(X_va) < 3:
        raise optuna.TrialPruned()

    hidden = trial.suggest_int('hidden', 64, 256, step=64)
    layers = trial.suggest_int('layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.2)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch = trial.suggest_categorical('batch', [16, 32, 64])

    model, history, baseline = train_lstm_model(
        X_tr, y_tr, X_va, y_va, device,
        hidden_size=hidden,
        num_layers=layers,
        dropout=dropout,
        lr=lr,
        epochs=60,
        batch_size=batch,
        patience=12,
        verbose=False  # Optuna时不显示详细输出
    )

    best_val = min(history['val_loss'])

    if best_val > baseline * 1.1:
        raise optuna.TrialPruned()

    return best_val


def visualize_training_history(history, indicator, output_dir):
    """可视化训练历史"""
    plt.figure(figsize=(10, 5))

    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, alpha=0.8)

    plt.title(f'Training History - {indicator}', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'{output_dir}/training_history_{indicator}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 训练历史图已保存: {filename}")


def visualize_prediction(train_dates, y_train, test_dates, y_test, y_pred,
                         metrics, params, indicator, output_dir):
    """可视化预测结果"""
    plt.figure(figsize=(14, 7))

    plt.plot(train_dates, y_train, marker='o', linewidth=2, markersize=3,
             color='green', label='Train Data (80%)', alpha=0.7)

    plt.plot(test_dates, y_test, marker='o', linewidth=2.5, markersize=5,
             color='steelblue', label='Actual (Test 20%)', alpha=0.9)

    plt.plot(test_dates, y_pred, marker='s', linewidth=2.5, markersize=5,
             color='orangered', label='Predicted (LSTM)', alpha=0.9, linestyle='--')

    textstr = (f"R² = {metrics['R2']:.4f}\n"
               f"RMSE = {metrics['RMSE']:.3f}\n"
               f"MAE = {metrics['MAE']:.3f}\n"
               f"\n🔥 参数:\n"
               f"SeqLen = {params['seq_len']}\n"
               f"Hidden = {params['hidden']}\n"
               f"Layers = {params['layers']}\n"
               f"LR = {params['lr']:.5f}\n"
               f"Dropout = {params['dropout']:.2f}")

    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

    plt.title(f'北京 - {indicator}\nLSTM预测（🔥 Fixed v4.0）',
              fontsize=13, fontweight='bold')
    plt.xlabel('Date (2023, 按日聚合)', fontsize=11)
    plt.ylabel('Value', fontsize=11)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f'{output_dir}/prediction_{indicator}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 预测结果图已保存: {filename}")


def debug_single_indicator(beijing_df, indicator):
    """调试单个指标"""
    print(f"\n{'='*80}")
    print(f"🔍 调试指标: {indicator}")
    print(f"{'='*80}")

    # 按日聚合
    result = build_daily_series_debug(beijing_df, indicator, OUTPUT_DIR)

    if result is None:
        print(f"  ❌ 数据不足，跳过")
        return None

    dates, y_train_sc, y_test_sc, y_train, y_test, scaler = result
    y_train_sc = np.array(y_train_sc)
    y_test_sc = np.array(y_test_sc)

    # Optuna调参
    print(f"\n  [1/2] Optuna自动调参...")
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )

    study.optimize(
        lambda trial: objective(trial, y_train_sc, y_test_sc, DEVICE),
        n_trials=15,  # 调试用更少trials
        show_progress_bar=False
    )

    best_params = study.best_params
    seq_len = best_params['seq_len']

    print(f"  ✓ 最佳参数: {best_params}")

    # 训练最终模型
    print(f"\n  [2/2] 训练最终模型...")
    X, y = create_sequences(y_train_sc, seq_len)
    X_test, y_test_seq = create_sequences(y_test_sc, seq_len)

    val_split = int(len(X) * 0.8)
    X_tr, X_va = X[:val_split], X[val_split:]
    y_tr, y_va = y[:val_split], y[val_split:]

    final_model, history, baseline = train_lstm_model(
        X_tr, y_tr, X_va, y_va, DEVICE,
        hidden_size=best_params['hidden'],
        num_layers=best_params['layers'],
        dropout=best_params['dropout'],
        lr=best_params['lr'],
        epochs=60,
        batch_size=best_params['batch'],
        patience=12,
        verbose=True  # 显示详细训练过程
    )

    # 可视化训练历史
    visualize_training_history(history, indicator, OUTPUT_DIR)

    # 预测
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_test_pred = final_model(X_test_tensor).squeeze().cpu().numpy()

    # 反归一化
    y_test_orig = scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # 日期
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx + seq_len:]

    # 评估
    metrics = calculate_metrics(y_test_orig, y_test_pred_orig)

    print(f"\n  📊 最终评估结果:")
    print(f"    R² = {metrics['R2']:.4f}")
    print(f"    RMSE = {metrics['RMSE']:.3f}")
    print(f"    MAE = {metrics['MAE']:.3f}")

    # 可视化预测结果
    visualize_prediction(
        train_dates, y_train,
        test_dates, y_test_orig, y_test_pred_orig,
        metrics, best_params, indicator, OUTPUT_DIR
    )

    return metrics


def main():
    """主函数"""
    print("="*80)
    print("🔍 北京省LSTM调试脚本")
    print("="*80)

    # 加载数据
    print("\n[1/3] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')

    # 只选择北京
    beijing_df = df[df['Province'] == 'Beijing'].copy()
    beijing_df['Date'] = pd.to_datetime(beijing_df['Date'])

    print(f"✓ 北京省数据加载完成")
    print(f"  总记录数: {len(beijing_df)}")
    print(f"  监测站数: {beijing_df['Monitoring_Station'].nunique()}")
    print(f"  日期范围: {beijing_df['Date'].min()} 至 {beijing_df['Date'].max()}")

    # 只调试2个指标
    print(f"\n[2/3] 训练模型（仅2个指标用于调试）...")
    indicators = ['pH', 'Water_Temperature_C']

    results = {}
    for indicator in indicators:
        result = debug_single_indicator(beijing_df, indicator)
        if result:
            results[indicator] = result

    # 总结
    print(f"\n[3/3] 调试总结")
    print("="*80)
    if results:
        for indicator, metrics in results.items():
            print(f"  {indicator}:")
            print(f"    R² = {metrics['R2']:.4f}")
            print(f"    RMSE = {metrics['RMSE']:.3f}")
    else:
        print("  ⚠️ 没有成功的预测")

    print(f"\n✓ 所有输出已保存至: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
