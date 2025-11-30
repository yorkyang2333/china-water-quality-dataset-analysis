"""
🔥🔥🔥 北京省LSTM调试脚本 v7.0 - 增强版优化
Debug Script v7.0 - Enhanced Optimizations

🔥🔥🔥 v7.0 新增优化：
1. ✅ 外生因子也添加 lag1/lag7/ma7 → 喂足记忆
2. ✅ 残差连接 (Residual Connection) → 学变化而非绝对值
3. ✅ Seasonal Naive baseline → 更强基线对比
4. ✅ ReduceLROnPlateau 调度器 + AMP 混合精度 → 更快更稳
5. ✅ RobustScaler for features → 应对重尾分布

v6.0 的修复（保留）：
- ✅ 目标指标的历史值放回输入
- ✅ 添加lag1/lag7特征
- ✅ 修正持久性baseline
- ✅ 7日滚动均值
- ✅ 多变量输入
- ✅ 季节性特征
- ✅ ACF分析

使用方法：
python debug_beijing_v7_enhanced.py

日期：2025-10-31
版本：v7.0 (Enhanced Optimizations)
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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
        # 🔥 v7.0: Enable cuDNN benchmark for speedup
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ 使用Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print(f"✓ 使用CPU")
    return device

DEVICE = get_device()

OUTPUT_DIR = 'Outputs/DEBUG_Beijing_v7_Enhanced'
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
    """ACF分析可视化"""
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
    🔥🔥🔥 v7.0 增强版：多变量日序列构建

    新增优化：
    1. ✅ 给所有外生因子添加 lag1/lag7/ma7
    2. ✅ RobustScaler for features (应对重尾分布)
    3. ✅ MinMaxScaler for target (保持原样)

    参数:
    - province_df: 省份数据
    - target_indicator: 目标指标（要预测的）
    - feature_indicators: 特征指标列表（作为输入）

    返回:
    - dates, X_train_sc, X_test_sc, y_train_sc, y_test_sc, y_train, y_test,
      scaler_X, scaler_y, target_col_idx
    """
    df = province_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2023]

    # 按日聚合所有指标
    all_indicators = list(set([target_indicator] + feature_indicators))
    daily = (df.groupby('Date')[all_indicators]
               .mean()
               .sort_index())

    # 7日滚动均值平滑
    daily_smoothed = daily.rolling(window=7, min_periods=1, center=False).mean()

    # 连续日期 + 插值
    daily_smoothed = (daily_smoothed
                     .asfreq('D')
                     .interpolate(method='linear', limit_direction='both'))

    # 添加季节性特征
    day_of_year = daily_smoothed.index.dayofyear
    daily_smoothed['sin_season'] = np.sin(2 * np.pi * day_of_year / 365)
    daily_smoothed['cos_season'] = np.cos(2 * np.pi * day_of_year / 365)

    # 🔥🔥🔥 v7.0 新增：给目标添加 lag 特征
    daily_smoothed[f'{target_indicator}_lag1'] = daily_smoothed[target_indicator].shift(1)
    daily_smoothed[f'{target_indicator}_lag7'] = daily_smoothed[target_indicator].shift(7)

    # 🔥🔥🔥 v7.0 新增：给所有外生因子也添加 lag1/lag7/ma7
    for col in feature_indicators:
        daily_smoothed[f'{col}_lag1'] = daily_smoothed[col].shift(1)
        daily_smoothed[f'{col}_lag7'] = daily_smoothed[col].shift(7)
        daily_smoothed[f'{col}_ma7'] = daily_smoothed[col].rolling(7, min_periods=1).mean()

    # 去掉前7天（lag7产生的NaN）
    daily_smoothed = daily_smoothed.dropna()

    # 检查数据量
    if len(daily_smoothed) < 60:
        return None

    # 🔥🔥🔥 v7.0：构建完整特征列表
    # 目标：当前值 + lag1 + lag7
    target_features = [target_indicator,
                      f'{target_indicator}_lag1',
                      f'{target_indicator}_lag7']

    # 外生因子：当前值 + lag1 + lag7 + ma7
    exog_features = []
    for col in feature_indicators:
        exog_features.extend([
            col,
            f'{col}_lag1',
            f'{col}_lag7',
            f'{col}_ma7'
        ])

    # 季节特征
    season_features = ['sin_season', 'cos_season']

    # 完整特征列表
    feature_cols = target_features + exog_features + season_features

    print(f"    🔥 v7.0 特征增强: 目标3列 + 外生{len(exog_features)}列 + 季节2列 = {len(feature_cols)}列")

    # 时间切分
    split_idx = int(len(daily_smoothed) * 0.8)

    X_train = daily_smoothed[feature_cols].iloc[:split_idx].values
    X_test = daily_smoothed[feature_cols].iloc[split_idx:].values

    y_train = daily_smoothed[target_indicator].iloc[:split_idx].values
    y_test = daily_smoothed[target_indicator].iloc[split_idx:].values

    # 🔥🔥🔥 v7.0：分别缩放
    # Features: RobustScaler (应对重尾分布)
    # Target: MinMaxScaler (保持原样)
    scaler_X = RobustScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # 目标列索引（用于残差连接）
    target_col_idx = 0  # 目标列总是第一列

    return (daily_smoothed.index,
            X_train_sc, X_test_sc,
            y_train_sc, y_test_sc,
            y_train, y_test,
            scaler_X, scaler_y,
            target_col_idx,
            feature_cols)

class MultivariateLSTMModel(nn.Module):
    """
    🔥🔥🔥 v7.0 增强版：带残差连接的多变量LSTM

    新增：
    - target_col_idx: 目标列在特征中的索引
    - 残差连接：输出 = last_target + delta
    """
    def __init__(self, input_size, target_col_idx=0, hidden_size=128, num_layers=2, dropout=0.15):
        super(MultivariateLSTMModel, self).__init__()

        self.target_col_idx = target_col_idx

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
        """
        🔥🔥🔥 v7.0 残差连接

        x: [B, T, F]
        输出：last_target + delta
        """
        # 取最后一个时间步的目标列值
        last_target = x[:, -1, self.target_col_idx].unsqueeze(-1)  # [B, 1]

        # LSTM预测
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # [B, H]
        out = self.norm(out)

        # 预测增量 delta
        delta = self.head(out)  # [B, 1]

        # 🔥 残差连接：y_pred = y[t] + delta
        return last_target + delta

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_multivariate_sequences(X_data, y_data, seq_length):
    """创建多变量序列"""
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i+seq_length])
        y_seq.append(y_data[i+seq_length])

    return np.array(X_seq), np.array(y_seq)

def calculate_naive_baseline(y_train, y_val):
    """Naive baseline: y[t] = y[t-1]"""
    y_val_pred = np.roll(y_val, 1)
    y_val_pred[0] = y_train[-1] if len(y_train) > 0 else y_val[0]
    return mean_squared_error(y_val, y_val_pred)

def calculate_ma7_baseline(y_train, y_val):
    """7日滚动均值baseline"""
    if len(y_train) < 7:
        return float('inf')

    ma7_pred = np.mean(y_train[-7:])
    y_val_pred = np.full_like(y_val, ma7_pred)
    return mean_squared_error(y_val, y_val_pred)

def calculate_seasonal_naive(y_train, y_val):
    """
    🔥🔥🔥 v7.0 新增：Seasonal Naive baseline

    y[t] = y[t-7]
    """
    if len(y_val) < 7:
        return float('inf')

    y_val_pred = np.roll(y_val, 7)
    # 前7个用训练集最后7个
    if len(y_train) >= 7:
        y_val_pred[:7] = y_train[-7:]
    else:
        y_val_pred[:7] = y_train[-1] if len(y_train) > 0 else y_val[0]

    return mean_squared_error(y_val, y_val_pred)

def train_lstm_model(X_train, y_train, X_val, y_val, device, target_col_idx=0,
                     hidden_size=128, num_layers=2, dropout=0.15,
                     lr=0.001, epochs=60, batch_size=32, patience=12,
                     verbose=True):
    """
    🔥🔥🔥 v7.0 增强版训练

    新增：
    1. ReduceLROnPlateau 调度器
    2. AMP 混合精度训练
    3. Seasonal Naive baseline
    """

    # 计算多个baseline
    naive_mse = calculate_naive_baseline(y_train, y_val)
    ma7_mse = calculate_ma7_baseline(y_train, y_val)
    seasonal_mse = calculate_seasonal_naive(y_train, y_val)

    if verbose:
        print(f"      Naive baseline MSE: {naive_mse:.6f}")
        print(f"      MA7 baseline MSE: {ma7_mse:.6f}")
        print(f"      🔥 v7.0 Seasonal Naive MSE: {seasonal_mse:.6f}")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    # 🔥 v7.0: 添加 pin_memory 和 num_workers 提速
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=(device.type=='cuda'), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           pin_memory=(device.type=='cuda'), num_workers=0)

    # 获取input_size
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else 1

    model = MultivariateLSTMModel(
        input_size=input_size,
        target_col_idx=target_col_idx,  # 🔥 v7.0: 传入目标列索引
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 🔥🔥🔥 v7.0: ReduceLROnPlateau 替代 CosineAnnealing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4,
        min_lr=1e-5
    )

    # 🔥🔥🔥 v7.0: AMP 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 🔥 v7.0: 更高效

            # 🔥🔥🔥 v7.0: AMP forward
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)

            # 🔥🔥🔥 v7.0: AMP backward
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    y_pred = model(X_batch).squeeze()
                    loss = criterion(y_pred, y_batch)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 🔥🔥🔥 v7.0: ReduceLROnPlateau 基于 val_loss 调整
        scheduler.step(val_loss)

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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"      Epoch {epoch+1:3d}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")

    # 🔥🔥🔥 v7.0: 使用最佳的 baseline
    best_baseline = min(naive_mse, ma7_mse, seasonal_mse)
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

def objective(trial, X_train_sc, y_train_sc, device, target_col_idx):
    """Optuna目标函数（v7.0增强版）"""
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

    # 🔥 v7.0: 更合理的搜索空间
    hidden = trial.suggest_categorical('hidden', [96, 128, 192])
    layers = trial.suggest_int('layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.05, 0.2)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch = trial.suggest_categorical('batch', [16, 32, 64])

    model, history, baseline = train_lstm_model(
        X_tr, y_tr, X_va, y_va, device,
        target_col_idx=target_col_idx,  # 🔥 v7.0
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

    # 🔥 v7.0: 收紧剪枝阈值（30% 而非 50%）
    if best_val > baseline * 1.3:
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
             color='orangered', label='Predicted (LSTM v7.0)', alpha=0.9, linestyle='--')

    textstr = (f"🔥🔥🔥 v7.0增强优化:\n"
               f"✓ 外生因子lag/ma7\n"
               f"✓ 残差连接 (Δ预测)\n"
               f"✓ Seasonal Naive baseline\n"
               f"✓ ReduceLROnPlateau\n"
               f"✓ AMP混合精度\n"
               f"✓ RobustScaler\n"
               f"\n📊 评估:\n"
               f"R² = {metrics['R2']:.4f}\n"
               f"RMSE = {metrics['RMSE']:.3f}\n"
               f"MAE = {metrics['MAE']:.3f}\n"
               f"ACF Max = {max_acf:.3f}\n"
               f"\n⚙️ 参数:\n"
               f"SeqLen = {params['seq_len']}\n"
               f"Hidden = {params['hidden']}\n"
               f"Features = {params['n_features']}")

    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8.5,
             va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

    ax1.set_title(f'北京 - {indicator}\n🔥🔥🔥 v7.0 增强版LSTM预测',
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
    """调试单个指标（v7.0增强版）"""
    print(f"\n{'='*80}")
    print(f"🔍 目标指标: {target_indicator}")
    print(f"🔍 特征指标: {feature_indicators}")
    print(f"{'='*80}")

    # ACF分析（原始日均序列）
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
    print(f"\n  [1/3] 构建多变量序列（v7.0增强特征）...")
    result = build_multivariate_daily_series(beijing_df, target_indicator, feature_indicators)

    if result is None:
        print(f"  ❌ 数据不足")
        return None

    (dates, X_train_sc, X_test_sc, y_train_sc, y_test_sc,
     y_train, y_test, scaler_X, scaler_y, target_col_idx, feature_cols) = result

    print(f"    特征维度: {X_train_sc.shape[1]}")
    print(f"    训练样本: {len(X_train_sc)}")
    print(f"    测试样本: {len(X_test_sc)}")

    # Optuna调参
    print(f"\n  [2/3] Optuna自动调参（v7.0搜索空间）...")
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )

    study.optimize(
        lambda trial: objective(trial, X_train_sc, y_train_sc, DEVICE, target_col_idx),
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
    print(f"\n  [3/3] 训练最终模型（v7.0增强训练）...")
    X_seq, y_seq = create_multivariate_sequences(X_train_sc, y_train_sc, seq_len)
    X_test_seq, y_test_seq = create_multivariate_sequences(X_test_sc, y_test_sc, seq_len)

    val_split = int(len(X_seq) * 0.8)
    X_tr, X_va = X_seq[:val_split], X_seq[val_split:]
    y_tr, y_va = y_seq[:val_split], y_seq[val_split:]

    final_model, history, baseline = train_lstm_model(
        X_tr, y_tr, X_va, y_va, DEVICE,
        target_col_idx=target_col_idx,
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

    # 计算所有baseline
    naive_pred_orig = np.roll(y_test_orig, 1)
    naive_pred_orig[0] = y_train[-1]

    ma7_pred_orig = np.full_like(y_test_orig, np.mean(y_train[-7:]))

    # 🔥🔥🔥 v7.0: Seasonal Naive
    seasonal_pred_orig = np.roll(y_test_orig, 7)
    if len(y_train) >= 7:
        seasonal_pred_orig[:7] = y_train[-7:]
    else:
        seasonal_pred_orig[:7] = y_train[-1]

    naive_r2 = r2_score(y_test_orig, naive_pred_orig)
    ma7_r2 = r2_score(y_test_orig, ma7_pred_orig)
    seasonal_r2 = r2_score(y_test_orig, seasonal_pred_orig)

    print(f"\n  📊 最终评估:")
    print(f"    LSTM模型:")
    print(f"      R² = {metrics['R2']:.4f}")
    print(f"      RMSE = {metrics['RMSE']:.3f}")
    print(f"      MAE = {metrics['MAE']:.3f}")
    print(f"\n    Baseline对比:")
    print(f"      Naive (y[t]=y[t-1]): R² = {naive_r2:.4f}")
    print(f"      MA7 (7日均值): R² = {ma7_r2:.4f}")
    print(f"      🔥 v7.0 Seasonal Naive (y[t]=y[t-7]): R² = {seasonal_r2:.4f}")

    best_baseline_r2 = max(naive_r2, ma7_r2, seasonal_r2)
    print(f"\n    结论: LSTM {'超过' if metrics['R2'] > best_baseline_r2 else '未超过'} 最佳baseline")

    # 可视化
    best_params['n_features'] = len(feature_cols)
    visualize_comparison(
        train_dates, y_train,
        test_dates, y_test_orig, y_test_pred_orig,
        metrics, best_params, target_indicator, OUTPUT_DIR, max_acf
    )

    return metrics

def main():
    """主函数"""
    print("="*80)
    print("🔥🔥🔥 北京省LSTM调试脚本 v7.0 - 增强版优化")
    print("新增：外生因子lag/ma7 + 残差连接 + Seasonal Naive + ReduceLROnPlateau + AMP")
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
    print(f"\n[2/2] 训练模型（测试v7.0增强效果）...")

    results_summary = []

    # 示例1：预测溶解氧（DO）
    print("\n" + "="*80)
    print("示例1：预测溶解氧（v7.0增强版）")
    print("="*80)
    result1 = debug_single_indicator(
        beijing_df,
        target_indicator='Dissolved_Oxygen_mg_L',
        feature_indicators=['Water_Temperature_C', 'pH', 'Turbidity_NTU', 'Conductivity_uS_cm']
    )
    if result1:
        results_summary.append(('Dissolved_Oxygen_mg_L', result1['R2'], 'v7.0增强'))

    # 示例2：预测水温
    print("\n" + "="*80)
    print("示例2：预测水温（v7.0增强版）")
    print("="*80)
    result2 = debug_single_indicator(
        beijing_df,
        target_indicator='Water_Temperature_C',
        feature_indicators=['Dissolved_Oxygen_mg_L', 'pH']
    )
    if result2:
        results_summary.append(('Water_Temperature_C', result2['R2'], 'v7.0增强'))

    # 示例3：预测pH
    print("\n" + "="*80)
    print("示例3：预测pH（v7.0增强版）")
    print("="*80)
    result3 = debug_single_indicator(
        beijing_df,
        target_indicator='pH',
        feature_indicators=[]
    )
    if result3:
        results_summary.append(('pH', result3['R2'], 'v7.0增强'))

    # 生成汇总报告
    print(f"\n{'='*80}")
    print("📊 v7.0 预测效果汇总")
    print(f"{'='*80}")
    print(f"\n{'指标':<30} {'R²':>8} {'版本':<15}")
    print("-"*80)

    if results_summary:
        # 按R²排序
        results_summary.sort(key=lambda x: x[1], reverse=True)

        for indicator, r2, version in results_summary:
            indicator_name = indicator.replace('_', ' ')[:28]
            print(f"{indicator_name:<30} {r2:>8.4f} {version:<15}")

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
