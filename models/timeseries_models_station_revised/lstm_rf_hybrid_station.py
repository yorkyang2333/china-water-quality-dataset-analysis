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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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

class LSTMModel(nn.Module):
    """LSTM时间序列模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
    """训练LSTM"""
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

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

def main():
    """主程序"""
    print("\n" + "="*60)
    print("  LSTM + 随机森林混合模型 - 监测站级别")
    print("="*60)

    data = load_data()
    if data is None:
        print("❌ 数据加载失败")
        return

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data = data.dropna(subset=['Date'])

    indicators = ['pH', 'Dissolved_Oxygen_mg_L', 'COD_mg_L']
    results = []

    for indicator in indicators:  # 处理所有指标
        print(f"\n🎯 {indicator}")
        for station in data['Station'].unique():  # 处理所有站点
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

                # 训练模型
                lstm_model, scaler, lstm_cols = train_lstm(train, indicator, epochs=20)
                rf_model, rf_cols = train_rf(train, indicator)

                # 预测
                y_lstm = predict_lstm(lstm_model, test, indicator, scaler, lstm_cols)
                y_rf = predict_rf(rf_model, test, rf_cols)
                y_hybrid = 0.5 * y_lstm[:len(y_rf)] + 0.5 * y_rf

                # 评估
                y_true = test[indicator].values[-len(y_hybrid):]
                rmse = np.sqrt(mean_squared_error(y_true, y_hybrid))
                r2 = r2_score(y_true, y_hybrid)

                print(f"    ✓ RMSE={rmse:.4f}, R²={r2:.4f}")

                # 绘图
                plot_comparison(test['Date'].values, test[indicator].values,
                              y_lstm, y_rf, y_hybrid, indicator, station)

            except Exception as e:
                print(f"    ❌ {e}")

    print("\n✅ 完成!")
    print(f"📊 输出: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
