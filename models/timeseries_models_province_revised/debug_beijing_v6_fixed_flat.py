"""
🔥🔥🔥 北京省LSTM调试脚本 v6.0 - 修复"平线预测"问题
Debug Script v6.0 - Fix Flat Prediction Issue

🔥🔥🔥 关键修复（v6.0）：
1. ✅ 目标指标的历史值放回输入 → 恢复自回归能力
2. ✅ 添加lag1/lag7特征 → 显式提供近期历史
3. ✅ 修正持久性baseline → y[t]=y[t-1]而非常数
4. ✅ 7日滚动均值 → 降噪
5. ✅ 多变量输入 → 利用指标间相关性
6. ✅ 季节性特征 → sin/cos编码
7. ✅ ACF分析 → 可视化时间相关性

背景：
- 单变量时间序列的ACF太弱（lag-1 ≈ 0.08）
- 水质指标间存在相关性（如水温影响溶解氧）
- 7日滚动平滑可以提取趋势、降低噪声

使用方法：
python debug_beijing_v6_fixed_flat.py

测试内容：
- 6个不同类型的水质指标预测
- 验证v6.0修复在不同指标上的效果
- 生成汇总报告和R²排名

作者：Claude Code
日期：2025-10-31
版本：v6.0 (Fix Flat Prediction + 6 Indicators Test)
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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

torch.manual_seed(42)
np.random.seed(42)


def get_device():
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

OUTPUT_DIR = 'Outputs/DEBUG_Beijing_v6_Fixed_6Indicators'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/acf_plots", exist_ok=True)


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


def plot_acf_analysis(series, indicator, output_dir):
    """
    🔥 新增：ACF分析可视化

    显示自相关函数，判断序列是否有可预测的时序结构
    """
    plt.figure(figsize=(12, 5))

    # 计算ACF值
    acf_values = acf(series, nlags=30, fft=False)

    # 绘制ACF
    plt.subplot(1, 2, 1)
    plot_acf(series, lags=30, ax=plt.gca())
    plt.title(f'ACF Analysis - {indicator}', fontsize=11, fontweight='bold')
    plt.xlabel('Lag (days)', fontsize=9)
    plt.ylabel('Autocorrelation', fontsize=9)

    # 显示关键lag值
    plt.subplot(1, 2, 2)
    key_lags = [1, 7, 14, 30]
    lag_values = [acf_values[lag] for lag in key_lags if lag < len(acf_values)]
    plt.bar(range(len(lag_values)), lag_values, color='steelblue', alpha=0.7)
    plt.axhline(y=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='弱相关阈值(0.2)')
    plt.axhline(y=-0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xticks(range(len(lag_values)), [f'Lag-{lag}' for lag in key_lags[:len(lag_values)]])
    plt.ylabel('ACF Value', fontsize=9)
    plt.title('Key Lag ACF Values', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 添加诊断文本
    max_acf = max(abs(acf_values[1:min(31, len(acf_values))]))
    diagnosis = "强时序结构" if max_acf > 0.4 else "中等时序结构" if max_acf > 0.2 else "⚠️ 弱时序结构"
    plt.text(0.5, 0.95, f'诊断: {diagnosis}\n最大|ACF|: {max_acf:.3f}',
             transform=plt.gcf().transFigure, fontsize=10,
             bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
             ha='center', va='top')

    plt.tight_layout()

    filename = f'{output_dir}/acf_plots/acf_{indicator}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 ACF分析图: {filename}")
    print(f"    📊 ACF诊断: {diagnosis} (最大|ACF|={max_acf:.3f})")

    return max_acf


def build_multivariate_daily_series(province_df, target_indicator, feature_indicators):
    """
    🔥🔥🔥 关键修复v6：多变量日序列构建（含lag特征）

    改进：
    1. 7日滚动均值平滑 → 降噪
    2. ⭐⭐ 目标指标的历史值 + lag特征 → 恢复自回归能力
    3. 多个指标作为输入特征
    4. 季节性特征（sin/cos）

    参数:
    - province_df: 省份数据
    - target_indicator: 目标指标（要预测的）
    - feature_indicators: 特征指标列表（作为输入）

    返回:
    - dates, X_train_sc, X_test_sc, y_train_sc, y_test_sc, y_train, y_test, scaler_X, scaler_y
    """
    df = province_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2023]

    # 按日聚合所有指标
    all_indicators = list(set([target_indicator] + feature_indicators))
    daily = (df.groupby('Date')[all_indicators]
               .mean()
               .sort_index())

    # 🔥 7日滚动均值平滑
    daily_smoothed = daily.rolling(window=7, min_periods=1, center=False).mean()

    # 连续日期 + 插值
    daily_smoothed = (daily_smoothed
                     .asfreq('D')
                     .interpolate(method='linear', limit_direction='both'))

    # 🔥 添加季节性特征
    day_of_year = daily_smoothed.index.dayofyear
    daily_smoothed['sin_season'] = np.sin(2 * np.pi * day_of_year / 365)
    daily_smoothed['cos_season'] = np.cos(2 * np.pi * day_of_year / 365)

    # 🔥🔥🔥 关键修复：添加目标的lag特征（恢复自回归能力）
    daily_smoothed[f'{target_indicator}_lag1'] = daily_smoothed[target_indicator].shift(1)
    daily_smoothed[f'{target_indicator}_lag7'] = daily_smoothed[target_indicator].shift(7)

    # 去掉前7天（lag7产生的NaN）
    daily_smoothed = daily_smoothed.dropna()

    # 检查数据量
    if len(daily_smoothed) < 60:
        return None

    # 🔥🔥🔥 关键：特征列包含目标的当前值、lag值、其他指标、季节特征
    feature_cols = ([target_indicator,
                     f'{target_indicator}_lag1',
                     f'{target_indicator}_lag7'] +
                    feature_indicators +
                    ['sin_season', 'cos_season'])

    # 时间切分
    split_idx = int(len(daily_smoothed) * 0.8)

    X_train = daily_smoothed[feature_cols].iloc[:split_idx].values
    X_test = daily_smoothed[feature_cols].iloc[split_idx:].values

    y_train = daily_smoothed[target_indicator].iloc[:split_idx].values
    y_test = daily_smoothed[target_indicator].iloc[split_idx:].values

    # 分别缩放X和y，只用训练段
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return (daily_smoothed.index,
            X_train_sc, X_test_sc,
            y_train_sc, y_test_sc,
            y_train, y_test,
            scaler_X, scaler_y)


class MultivariateLSTMModel(nn.Module):
    """
    🔥 多变量LSTM模型

    改进：支持多个输入特征
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.15):
        super(MultivariateLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  # 现在>1
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


def create_multivariate_sequences(X_data, y_data, seq_length):
    """
    创建多变量序列

    X_data: [T, n_features]
    y_data: [T]

    返回:
    X_seq: [N, seq_length, n_features]
    y_seq: [N]
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i+seq_length])  # [seq_length, n_features]
        y_seq.append(y_data[i+seq_length])    # scalar

    return np.array(X_seq), np.array(y_seq)


def calculate_naive_baseline(y_train, y_val):
    """Naive baseline: y[t] = y[t-1]"""
    y_val_pred = np.roll(y_val, 1)
    y_val_pred[0] = y_train[-1] if len(y_train) > 0 else y_val[0]
    return mean_squared_error(y_val, y_val_pred)


def calculate_ma7_baseline(y_train, y_val):
    """
    🔥 新增：7日滚动均值baseline

    用训练集最后7天的均值预测整个验证集
    """
    if len(y_train) < 7:
        return float('inf')

    ma7_pred = np.mean(y_train[-7:])
    y_val_pred = np.full_like(y_val, ma7_pred)
    return mean_squared_error(y_val, y_val_pred)


def train_lstm_model(X_train, y_train, X_val, y_val, device,
                     hidden_size=128, num_layers=2, dropout=0.15,
                     lr=0.001, epochs=60, batch_size=32, patience=12,
                     verbose=True):
    """训练多变量LSTM"""

    # 计算多个baseline
    naive_mse = calculate_naive_baseline(y_train, y_val)
    ma7_mse = calculate_ma7_baseline(y_train, y_val)

    if verbose:
        print(f"      Naive baseline MSE: {naive_mse:.6f}")
        print(f"      MA7 baseline MSE: {ma7_mse:.6f}")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 获取input_size
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else 1

    model = MultivariateLSTMModel(
        input_size=input_size,
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

    # baseline比较（仅用于信息显示）
    best_baseline = min(naive_mse, ma7_mse)
    baseline_ratio = best_val_loss / best_baseline
    if baseline_ratio > 1.2:
        print(f"      ⚠️ 模型显著差于baseline! (model={best_val_loss:.6f} vs baseline={best_baseline:.6f}, ratio={baseline_ratio:.2f})")
    elif baseline_ratio > 1.0:
        print(f"      ℹ️  模型略差于baseline (model={best_val_loss:.6f} vs baseline={best_baseline:.6f}, ratio={baseline_ratio:.2f})")
    else:
        print(f"      ✓ 模型超过baseline! (model={best_val_loss:.6f} vs baseline={best_baseline:.6f}, ratio={baseline_ratio:.2f})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_baseline


def objective(trial, X_train_sc, y_train_sc, device):
    """Optuna目标函数（多变量版）"""
    seq_len = trial.suggest_categorical('seq_len', [5, 10, 15, 20, 25, 30])

    X_seq, y_seq = create_multivariate_sequences(X_train_sc, y_train_sc, seq_len)

    if len(X_seq) < 30:
        raise optuna.TrialPruned()

    val_split = int(len(X_seq) * 0.8)
    if val_split < 15:
        raise optuna.TrialPruned()

    X_tr, y_tr = X_seq[:val_split], y_seq[:val_split]
    X_va, y_va = X_seq[val_split:], y_seq[val_split:]

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
        verbose=False
    )

    best_val = min(history['val_loss'])

    # 🔥 放宽守门：允许比baseline稍差（因为7日MA baseline很强）
    # 如果比baseline差50%以上才剪枝
    if best_val > baseline * 1.5:
        raise optuna.TrialPruned()

    return best_val


def visualize_comparison(train_dates, y_train, test_dates, y_test, y_pred,
                         metrics, params, indicator, output_dir, max_acf):
    """可视化预测结果"""
    fig = plt.figure(figsize=(14, 10))

    # 上图：预测对比
    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(train_dates, y_train, marker='o', linewidth=2, markersize=3,
             color='green', label='Train Data (80%, 7-day MA)', alpha=0.7)

    ax1.plot(test_dates, y_test, marker='o', linewidth=2.5, markersize=5,
             color='steelblue', label='Actual (Test 20%)', alpha=0.9)

    ax1.plot(test_dates, y_pred, marker='s', linewidth=2.5, markersize=5,
             color='orangered', label='Predicted (Multivariate LSTM)', alpha=0.9, linestyle='--')

    textstr = (f"🔥🔥🔥 v6.0关键修复:\n"
               f"✓ 目标历史值+lag1/lag7\n"
               f"✓ 7日滚动均值平滑\n"
               f"✓ 多变量输入 ({len(params.get('features', []))}个特征)\n"
               f"✓ 季节性特征 (sin/cos)\n"
               f"\n📊 评估:\n"
               f"R² = {metrics['R2']:.4f}\n"
               f"RMSE = {metrics['RMSE']:.3f}\n"
               f"MAE = {metrics['MAE']:.3f}\n"
               f"ACF Max = {max_acf:.3f}\n"
               f"\n⚙️ 参数:\n"
               f"SeqLen = {params['seq_len']}\n"
               f"Hidden = {params['hidden']}")

    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

    ax1.set_title(f'北京 - {indicator}\n多变量LSTM预测（🔥🔥🔥 v6.0 - 修复平线问题）',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date (2023)', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 下图：残差分布
    ax2 = plt.subplot(2, 1, 2)
    residuals = y_test - y_pred
    ax2.scatter(test_dates, residuals, alpha=0.6, color='purple', s=50)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.fill_between(test_dates, -2*np.std(residuals), 2*np.std(residuals),
                     alpha=0.2, color='gray', label='±2σ')

    ax2.set_title('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Residual', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    filename = f'{output_dir}/prediction_{indicator}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 预测图: {filename}")


def debug_single_indicator(beijing_df, target_indicator, feature_indicators):
    """调试单个指标（多变量版）"""
    print(f"\n{'='*80}")
    print(f"🔍 目标指标: {target_indicator}")
    print(f"🔍 特征指标: {feature_indicators}")
    print(f"{'='*80}")

    # 🔥 ACF分析（原始日均序列）
    df = beijing_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2023]
    daily_raw = (df.groupby('Date')[target_indicator]
                   .mean()
                   .sort_index()
                   .asfreq('D')
                   .interpolate(method='linear', limit_direction='both'))

    print(f"\n  [0/3] ACF分析（原始日均，未平滑）...")
    max_acf = plot_acf_analysis(daily_raw.values, target_indicator, OUTPUT_DIR)

    # 构建多变量序列
    print(f"\n  [1/3] 构建多变量序列（7日滚动+季节特征）...")
    result = build_multivariate_daily_series(beijing_df, target_indicator, feature_indicators)

    if result is None:
        print(f"  ❌ 数据不足")
        return None

    dates, X_train_sc, X_test_sc, y_train_sc, y_test_sc, y_train, y_test, scaler_X, scaler_y = result

    print(f"    特征维度: {X_train_sc.shape[1]}")
    print(f"    训练样本: {len(X_train_sc)}")
    print(f"    测试样本: {len(X_test_sc)}")

    # Optuna调参
    print(f"\n  [2/3] Optuna自动调参...")
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )

    study.optimize(
        lambda trial: objective(trial, X_train_sc, y_train_sc, DEVICE),
        n_trials=15,
        show_progress_bar=False
    )

    # 检查是否有完成的trial
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(completed_trials) == 0:
        print(f"  ⚠️ 所有trial都被剪枝，使用默认参数")
        best_params = {
            'seq_len': 15,
            'hidden': 128,
            'layers': 2,
            'dropout': 0.15,
            'lr': 0.001,
            'batch': 32
        }
    else:
        best_params = study.best_params
        print(f"  ✓ 最佳参数: {best_params}")

    seq_len = best_params['seq_len']

    # 训练最终模型
    print(f"\n  [3/3] 训练最终模型...")
    X_seq, y_seq = create_multivariate_sequences(X_train_sc, y_train_sc, seq_len)
    X_test_seq, y_test_seq = create_multivariate_sequences(X_test_sc, y_test_sc, seq_len)

    val_split = int(len(X_seq) * 0.8)
    X_tr, X_va = X_seq[:val_split], X_seq[val_split:]
    y_tr, y_va = y_seq[:val_split], y_seq[val_split:]

    final_model, history, baseline = train_lstm_model(
        X_tr, y_tr, X_va, y_va, DEVICE,
        hidden_size=best_params['hidden'],
        num_layers=best_params['layers'],
        dropout=best_params['dropout'],
        lr=best_params['lr'],
        epochs=60,
        batch_size=best_params['batch'],
        patience=12,
        verbose=True
    )

    # 预测
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
        y_test_pred = final_model(X_test_tensor).squeeze().cpu().numpy()

    # 反归一化
    y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # 日期
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx + seq_len:]

    # 评估
    metrics = calculate_metrics(y_test_orig, y_test_pred_orig)

    # 🔥🔥🔥 修正：真正的持久性baseline（y[t] = y[t-1]）
    naive_pred_orig = np.roll(y_test_orig, 1)
    naive_pred_orig[0] = y_train[-1]  # 第一个用训练集最后值

    ma7_pred_orig = np.full_like(y_test_orig, np.mean(y_train[-7:]))

    naive_r2 = r2_score(y_test_orig, naive_pred_orig)
    ma7_r2 = r2_score(y_test_orig, ma7_pred_orig)

    print(f"\n  📊 最终评估:")
    print(f"    LSTM模型:")
    print(f"      R² = {metrics['R2']:.4f}")
    print(f"      RMSE = {metrics['RMSE']:.3f}")
    print(f"      MAE = {metrics['MAE']:.3f}")
    print(f"\n    Baseline对比:")
    print(f"      Naive (y[t]=y[t-1]): R² = {naive_r2:.4f}")
    print(f"      MA7 (7日均值): R² = {ma7_r2:.4f}")
    print(f"\n    结论: LSTM {'超过' if metrics['R2'] > max(naive_r2, ma7_r2) else '未超过'} baseline")

    # 可视化
    best_params['features'] = feature_indicators + ['sin_season', 'cos_season']
    visualize_comparison(
        train_dates, y_train,
        test_dates, y_test_orig, y_test_pred_orig,
        metrics, best_params, target_indicator, OUTPUT_DIR, max_acf
    )

    return metrics


def main():
    """主函数"""
    print("="*80)
    print("🔥🔥🔥 北京省LSTM调试脚本 v6.0")
    print("关键修复：目标历史值+lag特征（解决平线预测）")
    print("="*80)

    # 加载数据
    print("\n[1/2] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')

    beijing_df = df[df['Province'] == 'Beijing'].copy()
    beijing_df['Date'] = pd.to_datetime(beijing_df['Date'])

    print(f"✓ 北京数据: {len(beijing_df)} 条")

    # 调试多个指标
    print(f"\n[2/2] 训练模型（共6个指标）...")

    results_summary = []

    # 🔥 示例1：预测溶解氧（DO），用水温、pH、浊度作为输入特征
    print("\n" + "="*80)
    print("示例1：预测溶解氧（多变量输入）")
    print("="*80)
    result1 = debug_single_indicator(
        beijing_df,
        target_indicator='Dissolved_Oxygen_mg_L',
        feature_indicators=['Water_Temperature_C', 'pH', 'Turbidity_NTU', 'Conductivity_uS_cm']
    )
    if result1:
        results_summary.append(('Dissolved_Oxygen_mg_L', result1['R2'], '多变量'))

    # 🔥 示例2：预测水温（强季节性）
    print("\n" + "="*80)
    print("示例2：预测水温（预期有明显季节性）")
    print("="*80)
    result2 = debug_single_indicator(
        beijing_df,
        target_indicator='Water_Temperature_C',
        feature_indicators=['Dissolved_Oxygen_mg_L', 'pH']  # 水温会影响DO
    )
    if result2:
        results_summary.append(('Water_Temperature_C', result2['R2'], '多变量+季节'))

    # 🔥 示例3：预测pH（仅季节特征）
    print("\n" + "="*80)
    print("示例3：预测pH（仅目标历史+季节特征）")
    print("="*80)
    result3 = debug_single_indicator(
        beijing_df,
        target_indicator='pH',
        feature_indicators=[]  # 仅使用目标历史+季节特征
    )
    if result3:
        results_summary.append(('pH', result3['R2'], '目标+季节'))

    # 🔥 示例4：预测总氮（营养盐指标）
    print("\n" + "="*80)
    print("示例4：预测总氮（营养盐指标）")
    print("="*80)
    result4 = debug_single_indicator(
        beijing_df,
        target_indicator='Total_Nitrogen_mg_L',
        feature_indicators=['Total_Phosphorus_mg_L', 'Ammonia_N_mg_L', 'Nitrate_mg_L']
    )
    if result4:
        results_summary.append(('Total_Nitrogen_mg_L', result4['R2'], '多营养盐'))

    # 🔥 示例5：预测浊度（物理指标）
    print("\n" + "="*80)
    print("示例5：预测浊度（物理指标）")
    print("="*80)
    result5 = debug_single_indicator(
        beijing_df,
        target_indicator='Turbidity_NTU',
        feature_indicators=['Water_Temperature_C', 'Conductivity_uS_cm']
    )
    if result5:
        results_summary.append(('Turbidity_NTU', result5['R2'], '物理指标'))

    # 🔥 示例6：预测COD（有机污染指标）
    print("\n" + "="*80)
    print("示例6：预测COD（有机污染指标）")
    print("="*80)
    result6 = debug_single_indicator(
        beijing_df,
        target_indicator='COD_mg_L',
        feature_indicators=['BOD_mg_L', 'Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L']
    )
    if result6:
        results_summary.append(('COD_mg_L', result6['R2'], '有机污染'))

    # 生成汇总报告
    print(f"\n{'='*80}")
    print("📊 v6.0 预测效果汇总")
    print(f"{'='*80}")
    print(f"\n{'指标':<30} {'R²':>8} {'类型':<15}")
    print("-"*80)

    if results_summary:
        # 按R²排序
        results_summary.sort(key=lambda x: x[1], reverse=True)

        for indicator, r2, type_str in results_summary:
            indicator_name = indicator.replace('_', ' ')[:28]
            print(f"{indicator_name:<30} {r2:>8.4f} {type_str:<15}")

        avg_r2 = sum(r[1] for r in results_summary) / len(results_summary)
        print("-"*80)
        print(f"{'平均 R²':<30} {avg_r2:>8.4f}")

        # 分类统计
        excellent = sum(1 for r in results_summary if r[1] > 0.7)
        good = sum(1 for r in results_summary if 0.5 <= r[1] <= 0.7)
        medium = sum(1 for r in results_summary if 0.3 <= r[1] < 0.5)
        poor = sum(1 for r in results_summary if r[1] < 0.3)

        print(f"\n效果分布:")
        print(f"  优秀 (R² > 0.7): {excellent} 个")
        print(f"  良好 (0.5-0.7): {good} 个")
        print(f"  中等 (0.3-0.5): {medium} 个")
        print(f"  较差 (< 0.3): {poor} 个")

    print(f"\n✓ 所有输出保存至: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
