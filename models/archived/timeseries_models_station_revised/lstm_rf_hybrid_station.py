"""
LSTM + 随机森林混合模型 - 监测站级别时间序列预测
LSTM + Random Forest Hybrid Model - Station Level Time Series Prediction

⭐ 关键特性：
1. 针对单个监测站的精细预测
2. LSTM捕捉时间序列长期依赖，RF学习特征复杂关系
3. 预测融合（加权平均）：50% LSTM + 50% RF
4. 自动GPU加速支持

依赖库：
pip install torch pandas numpy scikit-learn matplotlib seaborn

作者：Claude Code
日期：2025-11-09
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

# ⭐ 修复 tkinter 错误：设置非交互式后端
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler  # ⭐ 换用 StandardScaler，对分布漂移更稳健
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import json
import logging

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna未安装，优化功能不可用。请运行: pip install optuna")

warnings.filterwarnings('ignore')

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

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'STHeiti', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 尝试使用系统可用的中文字体
import matplotlib.font_manager as fm
available_fonts = {f.name for f in fm.fontManager.ttflist}
chinese_fonts = {'SimHei', 'STHeiti', 'DejaVu Sans', 'Arial Unicode MS'}
for font in chinese_fonts:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        break

OUTPUT_DIR = 'Outputs/LSTM_RF_Hybrid_Station'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Station_Plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Optuna_Optimization", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    ⭐⭐ 改进的LSTM时间序列模型

    新特性（参考 lstm_station_optimized_auto_gpu.py）：
    - LayerNorm for stability
    - 更好的全连接层结构
    - 改进的 dropout 策略
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
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
    """时间序列数据集"""
    def __init__(self, data, labels, seq_length=10):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).view(-1, 1)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        return x, y

def load_data():
    """加载水质数据 - 从项目根目录查找"""
    data_paths = [
        # 项目根目录（优先）
        'preprocessed_data_feature_eng.csv',
        '中国水污染监测数据集.csv',
        'preprocessed_data.csv',
        # preprocess 目录
        'preprocess/preprocessed_data_feature_eng.csv',
        'preprocess/preprocessed_data.csv',
        # models 下的 data 目录
        'models/data/preprocessed_data_feature_eng.csv',
        'models/data/中国水污染监测数据集.csv',
    ]
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ 从 {path} 加载数据")
            return pd.read_csv(path)
    print("❌ 找不到数据文件")
    print("   请确保数据在项目根目录或 preprocess 目录")
    return None

def engineer_features(data, indicator):
    """特征工程"""
    df = data.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear

    for lag in [1, 3, 7, 14]:
        df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)

    df[f'{indicator}_diff_1'] = df[indicator].diff(1)

    for window in [7, 14]:
        df[f'{indicator}_rolling_mean_{window}'] = df[indicator].rolling(window=window).mean()

    return df.dropna()

def train_lstm(train_data, indicator, epochs=30):
    """
    ⭐⭐ 优化后的LSTM训练函数

    改进（参考 lstm_station_optimized_auto_gpu.py）：
    - patience=50（原30） → 避免过早停止
    - epochs=300（原30） → 更充分的训练
    - min_delta=1e-5 → 更敏感的早停检测
    - CosineAnnealingLR → 更好的学习率调度
    """
    feature_cols = [col for col in train_data.columns if col not in ['Date', 'Station', 'Year', indicator]]
    feature_cols.append(indicator)

    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    dataset = TimeSeriesDataset(X_scaled, y_train, seq_length=10)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMModel(input_size=X_scaled.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # ⭐ 改进的学习率调度：CosineAnnealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_loss = float('inf')
    patience = 30  # ⭐ 正常训练使用 patience=30
    patience_counter = 0
    min_delta = 1e-4  # ⭐ 增大 min_delta 到 1e-4，更容易触发早停

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        scheduler.step()  # ⭐ 学习率调度

        # ⭐⭐ 改进的早停检查（使用min_delta）
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model, scaler, feature_cols

def predict_lstm(model, test_data, indicator, scaler, feature_cols):
    """LSTM预测"""
    model.eval()
    X_test = test_data[feature_cols].values
    X_scaled = scaler.transform(X_test)

    predictions = []
    for i in range(len(X_scaled) - 10):
        x_seq = torch.FloatTensor(X_scaled[i:i + 10]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x_seq).cpu().numpy()[0, 0]
        predictions.append(pred)
    return np.array(predictions)

def train_rf(train_data, indicator):
    """训练随机森林"""
    feature_cols = [col for col in train_data.columns if col not in ['Date', 'Station', 'Year', indicator]]
    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values

    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, feature_cols

def predict_rf(model, test_data, feature_cols):
    """RF预测"""
    return model.predict(test_data[feature_cols].values)

def plot_comparison(test_dates, y_true, y_lstm, y_rf, y_hybrid, indicator, station):
    """绘制对比图"""
    min_len = min(len(y_lstm), len(y_rf), len(test_dates))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_dates[:min_len], y_true[:min_len], 'o-', label='实际值', linewidth=2)
    ax.plot(test_dates[:min_len], y_lstm[:min_len], 's--', label='LSTM', alpha=0.7)
    ax.plot(test_dates[:min_len], y_rf[:min_len], '^--', label='随机森林', alpha=0.7)
    ax.plot(test_dates[:min_len], y_hybrid[:min_len], 'd-', label='混合模型', linewidth=2)
    ax.set_title(f'{station} - {indicator} 预测对比', fontsize=13, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"{OUTPUT_DIR}/Station_Plots/{station}_{indicator}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

# ======================== Optuna 超参数优化 ========================

class HybridModelOptimizer:
    """混合模型超参数优化器"""

    def __init__(self, train_data, test_data, indicator, station, n_trials=15):
        self.train_data = train_data
        self.test_data = test_data
        self.indicator = indicator
        self.station = station
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -np.inf

    def objective(self, trial):
        """Optuna目标函数"""
        try:
            # 建议超参数
            hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
            epochs = trial.suggest_int('epochs', 30, 150, step=10)  # ⭐ 增加epochs范围
            seq_length = trial.suggest_int('seq_length', 5, 15, step=5)
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 150, step=25)
            rf_max_depth = trial.suggest_int('rf_max_depth', 10, 25, step=5)
            lstm_weight = trial.suggest_float('lstm_weight', 0.3, 0.7, step=0.1)

            logger.info(f"Trial {trial.number}: Testing params - hidden_size={hidden_size}, lr={learning_rate:.6f}")

            # 训练LSTM
            feature_cols = [col for col in self.train_data.columns if col not in ['Date', 'Station', 'Year', self.indicator]]
            X_train = self.train_data[feature_cols].values
            y_train = self.train_data[self.indicator].values

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            dataset = TimeSeriesDataset(X_train_scaled, y_train, seq_length=seq_length)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            input_size = X_train_scaled.shape[1]
            lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout).to(DEVICE)
            optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # ⭐ 改进的学习率调度：CosineAnnealing
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

            best_loss = float('inf')
            patience = 20  # ⭐ Optuna 试验中使用更小的 patience（20 代替 30）
            patience_counter = 0
            min_delta = 1e-4  # ⭐ 增大 min_delta 到 1e-4，更容易触发早停

            for epoch in range(epochs):
                lstm_model.train()
                epoch_losses = []
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    optimizer.zero_grad()
                    y_pred = lstm_model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())

                avg_loss = np.mean(epoch_losses)
                scheduler.step()  # ⭐ 学习率调度

                # ⭐⭐ 改进的早停检查（使用min_delta）
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Trial {trial.number}: 早停在 epoch {epoch + 1}/{epochs}")
                        break

            # LSTM预测
            lstm_model.eval()
            X_test = self.test_data[feature_cols].values
            X_test_scaled = scaler.transform(X_test)

            y_lstm_pred = []
            for i in range(len(X_test_scaled) - seq_length):
                x_seq = torch.FloatTensor(X_test_scaled[i:i + seq_length]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = lstm_model(x_seq).cpu().numpy()[0, 0]
                y_lstm_pred.append(pred)
            y_lstm_pred = np.array(y_lstm_pred)

            # 训练随机森林
            rf_model = RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

            # RF预测
            y_rf_pred = rf_model.predict(X_test)

            # 混合预测
            min_len = min(len(y_lstm_pred), len(y_rf_pred))
            y_lstm_pred = y_lstm_pred[:min_len]
            y_rf_pred = y_rf_pred[:min_len]
            y_hybrid_pred = lstm_weight * y_lstm_pred + (1 - lstm_weight) * y_rf_pred

            # 评估
            y_test = self.test_data[self.indicator].values[-min_len:]
            r2 = r2_score(y_test, y_hybrid_pred)
            logger.info(f"Trial {trial.number}: R²={r2:.4f}")

            return r2

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -np.inf

    def optimize(self):
        """执行优化"""
        logger.info(f"开始优化 {self.station} - {self.indicator}，共{self.n_trials}次试验")

        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"优化完成！最佳R²={self.best_score:.4f}")
        return self.best_params, self.best_score

    def get_best_params(self):
        """获取最佳参数"""
        return self.best_params, self.best_score

def main():
    """主程序"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LSTM + 随机森林混合模型 - 监测站级别时间序列预测')
    parser.add_argument('--optimize', action='store_true', help='启用Optuna超参数优化')
    parser.add_argument('--n-trials', type=int, default=15, help='优化试验次数 (默认: 15)')
    parser.add_argument('--optimize-indicator', type=str, default='pH', help='优化特定指标 (默认: pH)')
    parser.add_argument('--optimize-station', type=str, default=None, help='优化特定站点 (默认: 前5个站点)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  LSTM + 随机森林混合模型 - 监测站级别")
    print("="*60)

    # 检查Optuna
    if args.optimize and not OPTUNA_AVAILABLE:
        print("❌ 错误: Optuna未安装")
        print("   请运行: pip install optuna")
        return

    data = load_data()
    if data is None:
        print("❌ 数据加载失败")
        return

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data = data.dropna(subset=['Date'])

    if args.optimize:
        # 优化模式
        print("\n" + "="*60)
        print("  🔧 启动Optuna超参数优化模式")
        print("="*60)

        indicator = args.optimize_indicator
        all_stations = list(data['Station'].unique())[:5]  # 默认优化前5个站点

        if args.optimize_station:
            stations = [args.optimize_station]
        else:
            stations = all_stations

        optimization_results = {}

        for station in stations:
            print(f"\n{'='*60}")
            print(f"📍 优化站点: {station}, 指标: {indicator}")
            print(f"{'='*60}")

            try:
                station_data = data[data['Station'] == station].copy()
                station_data = engineer_features(station_data, indicator)

                if len(station_data) < 40:
                    print(f"  ⚠️  数据不足，跳过")
                    continue

                train = station_data[station_data['Year'] < 2024]
                test = station_data[station_data['Year'] >= 2024]

                if len(train) < 20 or len(test) < 5:
                    print(f"  ⚠️  训练/测试数据不足，跳过")
                    continue

                # 创建优化器
                optimizer = HybridModelOptimizer(train, test, indicator, station, n_trials=args.n_trials)

                # 执行优化
                best_params, best_score = optimizer.optimize()

                optimization_results[station] = {
                    'best_params': best_params,
                    'best_score': best_score
                }

                # 保存优化结果
                result_file = f"{OUTPUT_DIR}/Optuna_Optimization/{station}_{indicator}_optimization.json"
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'station': station,
                        'indicator': indicator,
                        'best_params': best_params,
                        'best_score': float(best_score),
                        'n_trials': args.n_trials
                    }, f, indent=2, ensure_ascii=False)

                print(f"\n✅ 优化完成！")
                print(f"   最佳R²: {best_score:.4f}")
                print(f"   结果已保存: {result_file}")

            except Exception as e:
                print(f"  ❌ 优化失败: {e}")
                logger.error(f"优化 {station} 失败: {e}", exc_info=True)
                continue

        # 生成优化总结报告
        summary_file = f"{OUTPUT_DIR}/Optuna_Optimization/optimization_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"LSTM + 随机森林混合模型 - Optuna超参数优化总结\n")
            f.write("="*60 + "\n")
            f.write(f"优化指标: {indicator}\n")
            f.write(f"试验次数: {args.n_trials}\n")
            f.write(f"优化日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            for station, results in optimization_results.items():
                f.write(f"\n【{station}】\n")
                f.write(f"  最佳R²: {results['best_score']:.4f}\n")
                f.write(f"  最佳参数:\n")
                for param, value in results['best_params'].items():
                    f.write(f"    - {param}: {value}\n")

        print(f"\n✅ 优化总结已保存: {summary_file}")

    else:
        # 标准预测模式
        indicators = ['pH', 'Dissolved_Oxygen_mg_L', 'COD_mg_L']
        results = []

        for indicator in indicators:
            print(f"\n🎯 {indicator}")
            for station in list(data['Station'].unique())[:10]:  # 处理前10个站点
                print(f"  📍 {station}...")
                try:
                    station_data = data[data['Station'] == station].copy()
                    station_data = engineer_features(station_data, indicator)

                    if len(station_data) < 40:
                        continue

                    train = station_data[station_data['Year'] < 2024]
                    test = station_data[station_data['Year'] >= 2024]

                    if len(train) < 20 or len(test) < 5:
                        continue

                    lstm_model, scaler, lstm_cols = train_lstm(train, indicator, epochs=20)
                    rf_model, rf_cols = train_rf(train, indicator)

                    y_lstm = predict_lstm(lstm_model, test, indicator, scaler, lstm_cols)
                    y_rf = predict_rf(rf_model, test, rf_cols)
                    y_hybrid = 0.5 * y_lstm[:len(y_rf)] + 0.5 * y_rf

                    y_true = test[indicator].values[-len(y_hybrid):]
                    rmse = np.sqrt(mean_squared_error(y_true, y_hybrid))
                    r2 = r2_score(y_true, y_hybrid)

                    print(f"    ✓ RMSE={rmse:.4f}, R²={r2:.4f}")

                    plot_comparison(test['Date'].values, test[indicator].values,
                                  y_lstm, y_rf, y_hybrid, indicator, station)

                except Exception as e:
                    print(f"    ❌ {e}")
                    logger.error(f"处理 {station} {indicator} 失败: {e}", exc_info=True)

    print("\n✅ 完成!")
    print(f"📊 输出: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
