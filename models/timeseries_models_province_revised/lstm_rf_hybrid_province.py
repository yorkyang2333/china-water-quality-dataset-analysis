"""
LSTM + 随机森林混合模型 - 省份级别时间序列预测
LSTM + Random Forest Hybrid Model - Province Level Time Series Prediction

⭐ 关键特性：
1. 混合架构：LSTM + Random Forest 预测融合
   - LSTM用于捕捉时间序列的长期依赖关系
   - Random Forest用于学习特征间的非线性关系
   - 通过加权平均融合两个模型的预测结果

2. 数据处理：
   - 将180个监测站的数据聚合到10个省份
   - 按时间序列特性提取特征
   - 2023年数据用于训练，2024年数据用于验证

3. 特征工程：
   - 时间特征：月份、季度、星期几
   - 滞后特征：1, 3, 7, 14, 30天的历史值
   - 滚动统计：均值、标准差、最小/最大值
   - 差分特征：一阶和二阶差分

4. 自动GPU加速：
   - ✅ NVIDIA GPU (CUDA)
   - ✅ Apple Silicon (MPS)
   - ✅ CPU 通用

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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ⭐ 自动检测设备
def get_device():
    """自动检测并返回最佳可用设备 优先级: CUDA > MPS > CPU"""
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

# 配置绘图
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'STHeiti', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)

# 尝试使用系统可用的中文字体
import matplotlib.font_manager as fm
available_fonts = {f.name for f in fm.fontManager.ttflist}
chinese_fonts = {'SimHei', 'STHeiti', 'DejaVu Sans', 'Arial Unicode MS'}
for font in chinese_fonts:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        break
sns.set_style("whitegrid")

# 输出目录配置
OUTPUT_DIR = 'Outputs/LSTM_RF_Hybrid_Province'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Province_Plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Model_Comparison", exist_ok=True)
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

# ======================== LSTM 模型定义 ========================

class LSTMModel(nn.Module):
    """LSTM时间序列模型"""
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
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 取最后一个时间步
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

# ======================== 数据处理函数 ========================

def load_data():
    """加载水质数据 - 从项目根目录查找"""
    try:
        # 尝试从项目根目录和相关位置读取数据
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
                data = pd.read_csv(path)
                print(f"  数据大小: {data.shape[0]} 行, {data.shape[1]} 列")
                return data

        print("❌ 找不到数据文件")
        print("   已搜索的路径:")
        for path in data_paths:
            print(f"     - {path}")
        print("\n   请确保在以下位置之一有数据文件:")
        print("     1. 项目根目录: preprocessed_data_feature_eng.csv 或 中国水污染监测数据集.csv")
        print("     2. preprocess 目录: preprocess/preprocessed_data_feature_eng.csv")
        return None
    except Exception as e:
        print(f"❌ 读取数据出错: {e}")
        return None

def aggregate_to_province(data):
    """将监测站级别数据聚合到省份级别"""
    print("\n📊 正在聚合数据到省份级别...")

    # 转换日期格式
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    # 按省份和日期分组，计算平均值
    province_data = data.groupby(['Province', 'Date']).agg({
        'Water_Temperature_C': 'mean',
        'pH': 'mean',
        'Dissolved_Oxygen_mg_L': 'mean',
        'Conductivity_uS_cm': 'mean',
        'Turbidity_NTU': 'mean',
        'Nitrate_mg_L': 'mean',
        'Nitrite_mg_L': 'mean',
        'Ammonia_N_mg_L': 'mean',
        'Total_Phosphorus_mg_L': 'mean',
        'Total_Nitrogen_mg_L': 'mean',
        'COD_mg_L': 'mean',
        'BOD_mg_L': 'mean',
        'Heavy_Metals_Pb_ug_L': 'mean',
        'Heavy_Metals_Cd_ug_L': 'mean',
        'Heavy_Metals_Hg_ug_L': 'mean',
        'Coliform_Count_CFU_100mL': 'mean'
    }).reset_index()

    print(f"✓ 聚合完成，共 {len(province_data['Province'].unique())} 个省份")
    return province_data

def engineer_features(data, indicator):
    """特征工程"""
    df = data.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # 时间特征
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['Date'].dt.dayofweek >= 5

    # 滞后特征
    for lag in [1, 3, 7, 14, 30]:
        df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)

    # 差分特征
    df[f'{indicator}_diff_1'] = df[indicator].diff(1)
    df[f'{indicator}_diff_2'] = df[indicator].diff(2)

    # 滚动统计
    for window in [7, 14, 30]:
        df[f'{indicator}_rolling_mean_{window}'] = df[indicator].rolling(window=window).mean()
        df[f'{indicator}_rolling_std_{window}'] = df[indicator].rolling(window=window).std()

    # 删除NaN
    df = df.dropna()

    return df

def prepare_train_test_data(province_data, indicator, test_year=2024):
    """准备训练和测试数据"""
    data = province_data.copy()
    data['Year'] = data['Date'].dt.year

    # 特征工程
    data = engineer_features(data, indicator)

    # 分割训练和测试集
    train_data = data[data['Year'] < test_year]
    test_data = data[data['Year'] >= test_year]

    if len(train_data) < 30 or len(test_data) < 5:
        return None

    return train_data, test_data, data

# ======================== LSTM 训练 ========================

def train_lstm_model(train_data, indicator, epochs=50, seq_length=10, batch_size=16):
    """训练LSTM模型"""
    print(f"\n🔵 正在训练LSTM模型...")

    # 准备特征列表
    feature_cols = [col for col in train_data.columns if col not in ['Date', 'Province', 'Year', indicator]]
    feature_cols.append(indicator)

    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values

    # 归一化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 创建数据集
    dataset = TimeSeriesDataset(X_train_scaled, y_train, seq_length=seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    input_size = X_train_scaled.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        if patience_counter >= patience:
            print(f"  早停在Epoch {epoch + 1}")
            break

    return model, scaler, feature_cols

def predict_lstm(model, test_data, indicator, scaler, feature_cols, seq_length=10):
    """使用LSTM进行预测"""
    model.eval()

    X_test = test_data[feature_cols].values
    X_test_scaled = scaler.transform(X_test)

    predictions = []

    # 滑动窗口预测
    for i in range(len(X_test_scaled) - seq_length):
        x_seq = torch.FloatTensor(X_test_scaled[i:i + seq_length]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x_seq).cpu().numpy()[0, 0]
        predictions.append(pred)

    return np.array(predictions)

# ======================== 随机森林训练 ========================

def train_rf_model(train_data, indicator):
    """训练随机森林模型"""
    print(f"\n🟢 正在训练随机森林模型...")

    feature_cols = [col for col in train_data.columns if col not in ['Date', 'Province', 'Year', indicator]]

    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print(f"✓ 随机森林训练完成")

    return model, feature_cols

def predict_rf(model, test_data, indicator, feature_cols):
    """使用随机森林进行预测"""
    X_test = test_data[feature_cols].values
    predictions = model.predict(X_test)
    return predictions

# ======================== 融合和评估 ========================

def hybrid_predict(lstm_pred, rf_pred, lstm_weight=0.5):
    """融合LSTM和RF的预测结果"""
    # 确保两个预测长度一致
    min_len = min(len(lstm_pred), len(rf_pred))
    lstm_pred = lstm_pred[:min_len]
    rf_pred = rf_pred[:min_len]

    hybrid_pred = lstm_weight * lstm_pred + (1 - lstm_weight) * rf_pred
    return hybrid_pred

def calculate_metrics(y_true, y_pred, model_name=""):
    """计算评估指标"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"  {model_name}:")
    print(f"    - RMSE: {rmse:.4f}")
    print(f"    - MAE:  {mae:.4f}")
    print(f"    - R²:   {r2:.4f}")

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_predictions(test_data, y_true, y_lstm, y_rf, y_hybrid, indicator, province):
    """绘制预测结果对比图"""
    # 确保长度一致
    min_len = min(len(y_lstm), len(y_rf), len(y_hybrid), len(y_true) - 10)

    test_dates = test_data['Date'].values[-min_len:]
    y_true_plot = y_true[-min_len:]
    y_lstm_plot = y_lstm[:min_len]
    y_rf_plot = y_rf[:min_len]
    y_hybrid_plot = y_hybrid[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 绘制时间序列
    axes[0].plot(test_dates, y_true_plot, 'o-', label='实际值', linewidth=2, markersize=4, color='black')
    axes[0].plot(test_dates, y_lstm_plot, 's--', label='LSTM预测', alpha=0.7, color='blue')
    axes[0].plot(test_dates, y_rf_plot, '^--', label='随机森林预测', alpha=0.7, color='green')
    axes[0].plot(test_dates, y_hybrid_plot, 'd-', label='混合模型预测', alpha=0.8, linewidth=2, color='red')
    axes[0].set_xlabel('日期', fontsize=11)
    axes[0].set_ylabel(f'{indicator} (单位)', fontsize=11)
    axes[0].set_title(f'{province} - {indicator} 时间序列预测对比', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 绘制误差对比
    lstm_errors = np.abs(y_lstm_plot - y_true_plot)
    rf_errors = np.abs(y_rf_plot - y_true_plot)
    hybrid_errors = np.abs(y_hybrid_plot - y_true_plot)

    axes[1].bar(np.arange(len(test_dates))[::3], lstm_errors[::3], alpha=0.6, label='LSTM误差', width=0.25, color='blue')
    axes[1].bar(np.arange(len(test_dates))[::3] + 0.25, rf_errors[::3], alpha=0.6, label='RF误差', width=0.25, color='green')
    axes[1].bar(np.arange(len(test_dates))[::3] + 0.5, hybrid_errors[::3], alpha=0.6, label='混合误差', width=0.25, color='red')
    axes[1].set_xlabel('时间点', fontsize=11)
    axes[1].set_ylabel('绝对误差', fontsize=11)
    axes[1].set_title('预测误差对比', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/Province_Plots/{province}_{indicator}_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {save_path}")
    plt.close()

# ======================== Optuna 超参数优化 ========================

class HybridModelOptimizer:
    """混合模型超参数优化器"""

    def __init__(self, train_data, test_data, indicator, province, n_trials=20):
        self.train_data = train_data
        self.test_data = test_data
        self.indicator = indicator
        self.province = province
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -np.inf
        self.study = None

    def objective(self, trial):
        """Optuna目标函数"""
        try:
            # 建议超参数
            hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
            epochs = trial.suggest_int('epochs', 20, 100, step=10)
            seq_length = trial.suggest_int('seq_length', 5, 20, step=5)
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 200, step=50)
            rf_max_depth = trial.suggest_int('rf_max_depth', 10, 30, step=5)
            lstm_weight = trial.suggest_float('lstm_weight', 0.3, 0.7, step=0.1)

            logger.info(f"Trial {trial.number}: Testing params - hidden_size={hidden_size}, num_layers={num_layers}, lr={learning_rate:.6f}")

            # 训练LSTM
            feature_cols = [col for col in self.train_data.columns if col not in ['Date', 'Province', 'Year', self.indicator]]
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

            best_loss = float('inf')
            patience = 10
            patience_counter = 0

            for epoch in range(epochs):
                lstm_model.train()
                epoch_loss = 0

                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    optimizer.zero_grad()
                    y_pred = lstm_model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                epoch_loss /= len(loader)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            # LSTM预测
            y_lstm_pred = predict_lstm(lstm_model, self.test_data, self.indicator, scaler, feature_cols, seq_length)

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
            y_rf_pred = predict_rf(rf_model, self.test_data, self.indicator, feature_cols)

            # 混合预测
            y_hybrid_pred = hybrid_predict(y_lstm_pred, y_rf_pred, lstm_weight=lstm_weight)

            # 评估
            y_test = self.test_data[self.indicator].values
            metrics = calculate_metrics(y_test[-len(y_hybrid_pred):], y_hybrid_pred, "")

            if metrics is None:
                return -np.inf

            r2_score_val = metrics['R2']
            logger.info(f"Trial {trial.number}: R²={r2_score_val:.4f}")

            return r2_score_val

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -np.inf

    def optimize(self):
        """执行优化"""
        logger.info(f"开始优化 {self.province} - {self.indicator}，共{self.n_trials}次试验")

        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"优化完成！最佳R²={self.best_score:.4f}")
        return self.best_params, self.best_score

    def get_best_params(self):
        """获取最佳参数"""
        return self.best_params, self.best_score

# ======================== 主程序 ========================

def main():
    """主程序"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LSTM + 随机森林混合模型 - 省份级别时间序列预测')
    parser.add_argument('--optimize', action='store_true', help='启用Optuna超参数优化')
    parser.add_argument('--n-trials', type=int, default=20, help='优化试验次数 (默认: 20)')
    parser.add_argument('--optimize-indicator', type=str, default='COD_mg_L', help='优化特定指标 (默认: COD_mg_L)')
    parser.add_argument('--optimize-province', type=str, default=None, help='优化特定省份 (默认: 所有省份)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  LSTM + 随机森林混合模型 - 省份级别时间序列预测")
    print("="*70)

    # 检查Optuna
    if args.optimize and not OPTUNA_AVAILABLE:
        print("❌ 错误: Optuna未安装")
        print("   请运行: pip install optuna")
        return

    # 加载数据
    print("\n📥 正在加载数据...")
    data = load_data()
    if data is None:
        print("❌ 数据加载失败")
        return

    # 聚合到省份
    province_data = aggregate_to_province(data)

    if args.optimize:
        # 优化模式
        print("\n" + "="*70)
        print("  🔧 启动Optuna超参数优化模式")
        print("="*70)

        indicator = args.optimize_indicator
        provinces = [args.optimize_province] if args.optimize_province else province_data['Province'].unique()

        optimization_results = {}

        for province in provinces:
            print(f"\n{'='*70}")
            print(f"📍 优化省份: {province}, 指标: {indicator}")
            print(f"{'='*70}")

            try:
                province_subset = province_data[province_data['Province'] == province]

                result = prepare_train_test_data(province_subset, indicator)
                if result is None:
                    print(f"  ⚠️  数据不足，跳过")
                    continue

                train_data, test_data, all_data = result

                # 创建优化器
                optimizer = HybridModelOptimizer(train_data, test_data, indicator, province, n_trials=args.n_trials)

                # 执行优化
                best_params, best_score = optimizer.optimize()

                optimization_results[province] = {
                    'best_params': best_params,
                    'best_score': best_score
                }

                # 保存优化结果
                result_file = f"{OUTPUT_DIR}/Optuna_Optimization/{province}_{indicator}_optimization.json"
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'province': province,
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
                logger.error(f"优化 {province} 失败: {e}", exc_info=True)
                continue

        # 生成优化总结报告
        summary_file = f"{OUTPUT_DIR}/Optuna_Optimization/optimization_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"LSTM + 随机森林混合模型 - Optuna超参数优化总结\n")
            f.write("="*70 + "\n")
            f.write(f"优化指标: {indicator}\n")
            f.write(f"试验次数: {args.n_trials}\n")
            f.write(f"优化日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            for province, results in optimization_results.items():
                f.write(f"\n【{province}】\n")
                f.write(f"  最佳R²: {results['best_score']:.4f}\n")
                f.write(f"  最佳参数:\n")
                for param, value in results['best_params'].items():
                    f.write(f"    - {param}: {value}\n")

        print(f"\n✅ 优化总结已保存: {summary_file}")

    else:
        # 标准预测模式
        target_indicators = [
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
            'Conductivity_uS_cm', 'Turbidity_NTU', 'COD_mg_L', 'BOD_mg_L'
        ]

        all_results = {}

        for indicator in target_indicators:
            print(f"\n{'='*70}")
            print(f"🎯 处理指标: {indicator}")
            print(f"{'='*70}")

            province_results = {}

            for province in province_data['Province'].unique():
                print(f"\n  📍 省份: {province}")

                try:
                    province_subset = province_data[province_data['Province'] == province]

                    result = prepare_train_test_data(province_subset, indicator)
                    if result is None:
                        print(f"    ⚠️  数据不足，跳过")
                        continue

                    train_data, test_data, all_data = result
                    y_test = test_data[indicator].values

                    lstm_model, scaler, feature_cols = train_lstm_model(train_data, indicator, epochs=30)
                    y_lstm_pred = predict_lstm(lstm_model, test_data, indicator, scaler, feature_cols)

                    rf_model, rf_feature_cols = train_rf_model(train_data, indicator)
                    y_rf_pred = predict_rf(rf_model, test_data, indicator, rf_feature_cols)

                    y_hybrid_pred = hybrid_predict(y_lstm_pred, y_rf_pred, lstm_weight=0.5)

                    print(f"\n    📊 模型评估:")
                    lstm_metrics = calculate_metrics(y_test[-len(y_lstm_pred):], y_lstm_pred, "LSTM")
                    rf_metrics = calculate_metrics(y_test[-len(y_rf_pred):], y_rf_pred, "随机森林")
                    hybrid_metrics = calculate_metrics(y_test[-len(y_hybrid_pred):], y_hybrid_pred, "混合模型")

                    province_results[province] = {
                        'lstm': lstm_metrics,
                        'rf': rf_metrics,
                        'hybrid': hybrid_metrics
                    }

                    plot_predictions(test_data, y_test, y_lstm_pred, y_rf_pred, y_hybrid_pred, indicator, province)

                except Exception as e:
                    print(f"    ❌ 处理出错: {e}")
                    logger.error(f"处理 {province} {indicator} 失败: {e}", exc_info=True)
                    continue

            all_results[indicator] = province_results

        # 生成总结报告
        print("\n" + "="*70)
        print("  📋 总结报告")
        print("="*70)

        report_path = f"{OUTPUT_DIR}/hybrid_model_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LSTM + 随机森林混合模型 - 省份级别时间序列预测\n")
            f.write("="*70 + "\n\n")

            for indicator, province_results in all_results.items():
                f.write(f"\n【{indicator}】\n")
                f.write("-"*70 + "\n")

                for province, results in province_results.items():
                    f.write(f"\n  {province}:\n")
                    f.write(f"    LSTM: RMSE={results['lstm']['RMSE']:.4f}, MAE={results['lstm']['MAE']:.4f}, R²={results['lstm']['R2']:.4f}\n")
                    f.write(f"    RF:   RMSE={results['rf']['RMSE']:.4f}, MAE={results['rf']['MAE']:.4f}, R²={results['rf']['R2']:.4f}\n")
                    f.write(f"    混合: RMSE={results['hybrid']['RMSE']:.4f}, MAE={results['hybrid']['MAE']:.4f}, R²={results['hybrid']['R2']:.4f}\n")

        print(f"\n✅ 报告已保存: {report_path}")

    print(f"\n🎉 完成!")
    print(f"   输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
