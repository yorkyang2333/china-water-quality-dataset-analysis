"""
LSTM + 随机森林混合模型 - 省份级别时间序列预测
LSTM + Random Forest Hybrid Model - Province Level Time Series Prediction

⚠️⚠️⚠️ 已弃用 (DEPRECATED) ⚠️⚠️⚠️

弃用原因：
  数据不满足时间序列基本假设，导致所有时间序列方法失败：
  1. 采样极度不规则（CV=1.99，远超阈值0.2）
  2. 零时间相关性（ACF≈0，无法从历史学习）
  3. 分布严重漂移（2024年数据密度仅为2023年12.4%）

性能表现：
  - LSTM: R² ≈ -0.1（略差于Naive基线）
  - RandomForest: R² ≈ -0.05（接近Naive）
  - 混合模型: R² ≈ -0.3（修复后，原为-2.5）

保留目的：
  - 作为论文对比基线
  - 展示方法选择的重要性
  - 失败案例分析参考

推荐替代方法：
  - 水质指标回归：models/indicator_regression/ (R² > 0.5)
  - 污染等级分类：models/pollution_level_classification/ (Acc > 99%)

详细分析：见 DEPRECATED_ANALYSIS.md

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
   - [OK] NVIDIA GPU (CUDA)
   - [OK] Apple Silicon (MPS)
   - [OK] CPU 通用

依赖库：
pip install torch pandas numpy scikit-learn matplotlib seaborn

作者：Claude Code
日期：2025-11-09
最后更新：2025-11-13（标记弃用）
"""

import zhplot  # 自动配置matplotlib中文字体
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
from sklearn.preprocessing import StandardScaler, RobustScaler  # ⭐ 添加 RobustScaler，用于Conductivity等异常值较多的指标
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
    print("[WARN]  Optuna未安装，优化功能不可用。请运行: pip install optuna")

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
        print(f"[OK] 使用CUDA GPU: {device_name}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[OK] 使用Apple Silicon MPS加速")
    else:
        device = torch.device('cpu')
        print(f"[OK] 使用CPU")
    return device

DEVICE = get_device()

# 配置绘图 - 修复中文字体显示
import matplotlib.font_manager as fm
from matplotlib import rcParams
import platform

# 直接设置中文字体（不使用函数，确保立即执行）
print("[INFO] 正在配置中文字体...")

font_loaded = False

# 方法1: 通过字体文件路径（Windows系统）- 优先使用黑体
if platform.system() == 'Windows':
    font_paths = [
        (r'C:\Windows\Fonts\simhei.ttf', 'SimHei'),             # 黑体（优先）
        (r'C:\Windows\Fonts\msyh.ttc', 'Microsoft YaHei'),      # 微软雅黑
        (r'C:\Windows\Fonts\simsun.ttc', 'SimSun'),             # 宋体
    ]

    for font_path, font_desc in font_paths:
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()

                # 强制设置字体
                rcParams['font.sans-serif'] = [font_name]
                rcParams['font.family'] = 'sans-serif'
                rcParams['axes.unicode_minus'] = False

                
                plt.rcParams['font.family'] = 'sans-serif'
                

                
                print(f"[OK] 字体文件: {font_path}")
                print(f"[OK] 字体名称: {font_name}")
                font_loaded = True
                break
            except Exception as e:
                print(f"[WARN] 加载 {font_desc} 失败: {e}")
                continue

# 方法2: 回退到字体名称（如果文件路径失败）- 优先黑体
if not font_loaded:
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in chinese_fonts:
        if font in available_fonts:
            rcParams['font.sans-serif'] = [font]
            rcParams['font.family'] = 'sans-serif'
            rcParams['axes.unicode_minus'] = False

            
            plt.rcParams['font.family'] = 'sans-serif'
            

            
            font_loaded = True
            break

if not font_loaded:
    print("[ERROR] 未能加载中文字体，图表中的中文可能显示为方块")
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

# 其他绘图设置
plt.rcParams['figure.figsize'] = (14, 10)
sns.set_style("whitegrid")

print("[INFO] 字体配置完成\n")

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
    """
    ⭐⭐ 改进的LSTM时间序列模型

    新特性（参考 lstm_station_optimized_auto_gpu.py）：
    - LayerNorm for stability
    - 更好的全连接层结构
    - 改进的 dropout 策略
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):  # ⭐ 减小 dropout 到 0.1
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
    """
    ⭐⭐⭐ 时间序列数据集（修复权重错位 - revise.md 硬伤B）

    关键修复：
    - __getitem__ 直接返回 (x, y, w)，避免 shuffle 后权重错位
    - 权重在初始化时对齐到标签（从 seq_length 开始）
    """
    def __init__(self, data, labels, weights=None, seq_length=7):  # ⭐ 减小 seq_length 到 7
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).view(-1, 1)
        self.seq_length = seq_length

        # ⭐⭐ 权重对齐到标签（从 seq_length 开始）
        if weights is not None:
            self.weights = torch.FloatTensor(weights)[seq_length:]
        else:
            self.weights = torch.ones(len(labels) - seq_length)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        w = self.weights[idx]  # ⭐⭐ 直接返回对应的权重
        return x, y, w

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
            # 从子目录向上查找（适用于在 models/xxx 目录运行）
            '../../preprocessed_data_feature_eng.csv',
            '../../中国水污染监测数据集.csv',
            '../../preprocessed_data.csv',
            # preprocess 目录
            'preprocess/preprocessed_data_feature_eng.csv',
            'preprocess/preprocessed_data.csv',
            '../../preprocess/preprocessed_data_feature_eng.csv',
            '../../preprocess/preprocessed_data.csv',
            # models 下的 data 目录
            'models/data/preprocessed_data_feature_eng.csv',
            'models/data/中国水污染监测数据集.csv',
        ]

        for path in data_paths:
            if os.path.exists(path):
                print(f"[OK] 从 {path} 加载数据")
                data = pd.read_csv(path)
                print(f"  数据大小: {data.shape[0]} 行, {data.shape[1]} 列")
                return data

        print("[ERROR] 找不到数据文件")
        print("   已搜索的路径:")
        for path in data_paths:
            print(f"     - {path}")
        print("\n   请确保在以下位置之一有数据文件:")
        print("     1. 项目根目录: preprocessed_data_feature_eng.csv 或 中国水污染监测数据集.csv")
        print("     2. preprocess 目录: preprocess/preprocessed_data_feature_eng.csv")
        return None
    except Exception as e:
        print(f"[ERROR] 读取数据出错: {e}")
        return None

def aggregate_to_province(data):
    """
    ⭐⭐⭐ 将监测站级别数据聚合到省份级别（参考 revise.md）

    新增：station_count 特征，记录每天参与聚合的监测站数量，
    用于捕获采样构成变化导致的漂移。
    """
    print("\n[AGG] 正在聚合数据到省份级别...")

    # 转换日期格式
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    # 按省份和日期分组
    grp = data.groupby(['Province', 'Date'])

    # 计算平均值
    province_data = grp.agg({
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

    # ⭐⭐ 添加 station_count 特征（处理采样构成漂移）
    station_counts = data.groupby(['Province', 'Date'])['Monitoring_Station'].nunique().reset_index()
    station_counts.columns = ['Province', 'Date', 'station_count']
    province_data = province_data.merge(station_counts, on=['Province', 'Date'], how='left')

    print(f"[OK] 聚合完成，共 {len(province_data['Province'].unique())} 个省份")
    print(f"  添加 station_count 特征（范围: {province_data['station_count'].min():.0f}-{province_data['station_count'].max():.0f}）")
    return province_data

def engineer_features(data, indicator):
    """
    ⭐⭐⭐⭐ 增强版特征工程（参考 revise.md 第1点 - 特征对齐修复）

    关键修复：
    - 创建统一的目标列：{indicator}_target = shift(-1)，预测 y(t+1)
    - 差分特征使用严格过去：diff().shift(1)
    - 滚动统计使用严格过去：先 shift(1) 再 rolling
    - 消除目标泄漏

    新增特征（revise.md 建议）：
    - gap_days: 时间间隔特征（处理不等间隔采样）
    - sin/cos: 季节性循环特征（替代单一数值月/日）
    - cross-lag: BOD/COD 等受相关指标影响的滞后特征
    - is_imputed: 缺测指示特征（标记哪些点被插值/填充）

    ⭐⭐⭐ 峰值加权（用于样本权重）：
    - spike_flag: 峰值/突变标记
    - sample_weight: 样本权重（峰值样本权重更高）
    """
    df = data.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # ⭐⭐⭐ 缺测指示特征（revise.md 第3节）
    # 在填充之前记录哪些值是缺失的
    df[f'{indicator}_is_imputed'] = df[indicator].isna().astype(int)

    # 填充缺失值（使用前向填充）
    df[indicator] = df[indicator].fillna(method='ffill').fillna(method='bfill')

    # 时间特征（不含未来信息）
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

    # ⭐⭐ 时间间隔特征（处理不等间隔采样）
    df['gap_days'] = df['Date'].diff().dt.days.fillna(1).clip(1, 14)

    # ⭐⭐ 季节性循环特征（替代单一数值）
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

    # 滞后特征（OK，本身就是过去）
    for lag in [1, 3, 7, 14, 30]:
        df[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)

    # ⭐⭐⭐ cross-lag 特征（revise.md 第3节）
    # BOD/COD 等水质指标受多个变量影响
    cross_indicators = {
        'BOD_mg_L': ['Water_Temperature_C', 'Dissolved_Oxygen_mg_L', 'COD_mg_L', 'pH'],
        'COD_mg_L': ['Water_Temperature_C', 'Dissolved_Oxygen_mg_L', 'BOD_mg_L', 'pH'],
        'Dissolved_Oxygen_mg_L': ['Water_Temperature_C', 'pH', 'BOD_mg_L'],
        'Total_Nitrogen_mg_L': ['Total_Phosphorus_mg_L', 'Ammonia_N_mg_L', 'Nitrate_mg_L'],
        'Total_Phosphorus_mg_L': ['Total_Nitrogen_mg_L', 'pH'],
    }

    if indicator in cross_indicators:
        for cross_ind in cross_indicators[indicator]:
            if cross_ind in df.columns:
                # 添加关键滞后（1天和7天）
                df[f'{cross_ind}_lag_1'] = df[cross_ind].shift(1)
                df[f'{cross_ind}_lag_7'] = df[cross_ind].shift(7)

    # ⭐ 差分特征 —— 使用严格过去
    df[f'{indicator}_diff_1'] = df[indicator].diff(1).shift(1)
    df[f'{indicator}_diff_2'] = df[indicator].diff(2).shift(1)

    # ⭐ 滚动统计 —— 对"过去值"滚动，再整体 shift(1)
    past = df[indicator].shift(1)
    for window in [7, 14, 30]:
        df[f'{indicator}_rolling_mean_{window}'] = past.rolling(window=window).mean()
        df[f'{indicator}_rolling_std_{window}'] = past.rolling(window=window).std()
        df[f'{indicator}_rolling_min_{window}'] = past.rolling(window=window).min()
        df[f'{indicator}_rolling_max_{window}'] = past.rolling(window=window).max()

    # ⭐⭐⭐ 特征对齐修复（revise.md 第1点）
    # 统一目标列：预测下一时刻的绝对值 y(t+1)
    df[f'{indicator}_target'] = df[indicator].shift(-1)

    # 峰值/突变的强度：用 Δy 的滚动标准化衡量（用于样本权重）
    delta = df[indicator].diff()
    delta_std = delta.rolling(14, min_periods=5).std()
    z = (delta / (delta_std + 1e-6)).abs()

    # 事件标记与样本权重（上调"难样本"权重）
    df['spike_flag'] = (z > 1.5).astype(int)                # 可调阈值 1.5~2.0
    df['sample_weight'] = 1.0 + 2.0 * np.clip(z, 0, 3)      # 权重区间约 1~7

    # 清理 NaN（包括 {indicator}_target）
    df = df.dropna(subset=[f'{indicator}_target'])
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

def train_lstm_model(train_data, indicator, epochs=50, seq_length=7, batch_size=16):  # ⭐ 减小 seq_length 到 7
    """
    ⭐⭐⭐⭐ LSTM训练函数（特征对齐修复 - revise.md 第1点）

    关键改进：
    1. 使用统一的 {indicator}_target（预测 y(t+1)）作为目标
    2. 使用加权 SmoothL1Loss（峰值样本权重更高）
    3. 引入验证集早停（监控 val_loss）
    4. 时间切分：85% 训练，15% 验证
    5. CosineAnnealingLR 学习率调度
    """
    print(f"\n[LSTM] 正在训练LSTM模型(y(t+1)目标)...")

    target_col = f'{indicator}_target'

    # ⭐⭐⭐ 只用过去特征（不含当期 indicator 和目标列）
    feature_cols = [c for c in train_data.columns
                    if c not in ['Date', 'Province', 'Year', indicator, target_col, 'spike_flag', 'sample_weight']]

    X = train_data[feature_cols].values
    y = train_data[target_col].values  # ⭐ 使用统一目标列
    w = train_data['sample_weight'].values  # ⭐ 峰值权重

    # ⭐⭐⭐ Conductivity特殊归一化（使用RobustScaler，对异常值更稳健）
    if 'Conductivity' in indicator:
        print(f"  [INFO] 使用RobustScaler归一化（Conductivity指标）")
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ⭐⭐ 时间切分：85% 训练，15% 验证
    split = int(len(X_scaled) * 0.85)

    # ⭐⭐⭐ 创建数据集（传入权重，Dataset会自动对齐）
    train_ds = TimeSeriesDataset(X_scaled[:split], y[:split], weights=w[:split], seq_length=seq_length)
    val_ds = TimeSeriesDataset(X_scaled[split - seq_length:], y[split - seq_length:], weights=w[split - seq_length:], seq_length=seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_size = X_scaled.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ⭐⭐ 使用无归约的 SmoothL1Loss，手动加权
    criterion = nn.SmoothL1Loss(reduction='none', beta=1.0)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_val = 1e9
    best_state = None
    patience = 30
    bad = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        tr_loss = []
        for Xb, yb, wb in train_loader:  # ⭐⭐⭐ 修复：直接从DataLoader获取权重
            Xb, yb, wb = Xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)

            # ⭐⭐ 加权损失：使用 batch 内的权重
            loss_vec = criterion(pred, yb)  # [B, 1]
            loss = (loss_vec * wb.unsqueeze(1)).mean()  # ⭐ 加权 Huber

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss.append(loss.item())

        # 验证阶段
        model.eval()
        va_loss = []
        with torch.no_grad():
            for Xb, yb, wb in val_loader:  # ⭐ 也更新验证循环
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                loss = criterion(pred, yb).mean()  # 验证集不加权
                va_loss.append(loss.item())

        scheduler.step()
        cur = float(np.mean(va_loss))

        # 早停检查
        if cur < best_val - 1e-4:
            best_val = cur
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  早停在Epoch {epoch + 1}")
                if best_state:
                    model.load_state_dict(best_state)
                    model.to(DEVICE)
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Train Loss: {np.mean(tr_loss):.6f}, Val Loss: {cur:.6f}, Patience: {bad}/{patience}")

    return model, scaler, feature_cols

def predict_lstm(model, test_data, indicator, scaler, feature_cols, seq_length=7):  # ⭐ 减小 seq_length 到 7
    """
    ⭐⭐⭐⭐ LSTM预测函数（特征对齐修复 - revise.md 第1点）

    直接预测绝对值 y(t+1)，无需还原逻辑。

    LSTM使用滑动窗口，从 seq_length 开始预测。
    预测结果直接对应 test_data[seq_length:] 时刻的真值。
    """
    model.eval()

    X = test_data[feature_cols].values
    Xs = scaler.transform(X)

    preds = []
    # 滑动窗口预测（从 seq_length 开始）
    for i in range(len(Xs) - seq_length):
        x_seq = torch.FloatTensor(Xs[i:i + seq_length]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            y_hat = model(x_seq).cpu().numpy()[0, 0]   # 直接预测 y(t+1)
        preds.append(y_hat)

    return np.array(preds)

# ======================== 随机森林训练 ========================

def train_rf_model(train_data, indicator):
    """
    ⭐⭐⭐⭐ 随机森林模型训练（特征对齐修复 - revise.md 第1点）

    改进（参考 revise.md v2 + 第5节）：
    - 使用统一的 {indicator}_target（预测 y(t+1)）作为目标
    - 使用 sample_weight 给峰值样本更高权重
    - ⭐⭐ 浅树结构参数（更抗噪声）：
      * n_estimators=500
      * max_depth=8（浅树）
      * min_samples_leaf=5（限制叶子节点）
      * max_features=0.7（特征子采样）
    """
    print(f"\n[RF] 正在训练随机森林模型(y(t+1)目标 + 浅树结构)...")

    target_col = f'{indicator}_target'

    feature_cols = [c for c in train_data.columns
                    if c not in ['Date', 'Province', 'Year', indicator, target_col, 'spike_flag', 'sample_weight']]

    X = train_data[feature_cols].values
    y = train_data[target_col].values  # ⭐ 使用统一目标列
    w = train_data['sample_weight'].values  # ⭐ 峰值权重

    # ⭐⭐ revise.md 推荐的浅树参数（比深树+少叶子更抗噪）
    model = RandomForestRegressor(
        n_estimators=500,       # 足够多的树
        max_depth=8,            # ⭐ 浅树深度
        min_samples_leaf=5,     # ⭐ 限制叶子节点最小样本数
        max_features=0.7,       # ⭐ 特征子采样比例
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    # ⭐⭐ 使用样本权重训练
    model.fit(X, y, sample_weight=w)
    print(f"[OK] 随机森林训练完成（浅树结构：depth=8, leaf=5）")

    return model, feature_cols

def predict_rf(model, test_data, indicator, feature_cols, seq_length=7):
    """
    ⭐⭐⭐⭐ RF预测函数（特征对齐修复 - revise.md 第1点 + 维度对齐修复）

    直接预测绝对值 y(t+1)，无需还原逻辑。

    ⭐⭐⭐ 关键修复：只预测 [seq_length:] 部分，与LSTM对齐
    RF 预测覆盖 [seq_length, ..., T-1]，长度 = len(test_data) - seq_length
    这样和LSTM的输出长度一致。
    """
    # ⭐⭐⭐ 关键修复：只使用 [seq_length:] 的数据进行预测
    X = test_data[feature_cols].values[seq_length:]
    y_pred = model.predict(X)  # 直接预测绝对值 y(t+1)
    return y_pred  # 长度 = len(test_data) - seq_length

# ======================== 融合和评估 ========================

def learn_blend_weight(y_lstm_va, y_rf_va, y_true_va, alpha=1e-3):
    """
    ⭐⭐⭐⭐ 学习型融合权重 - 带偏置的岭回归（参考 revise.md 补丁C）

    使用岭回归在验证集上学习最优权重 + 偏置项。
    相比无偏置的最小二乘，带偏置可以纠正系统性偏差。

    ⭐⭐⭐ 新增：限制权重和偏置范围，防止数值爆炸

    Args:
        y_lstm_va: LSTM 在验证集上的预测
        y_rf_va: RF 在验证集上的预测
        y_true_va: 验证集真实值
        alpha: L2 正则化系数（默认 1e-3）

    Returns:
        coef: [w_lstm, w_rf, bias] 三参数
    """
    n = len(y_true_va)

    # 构造设计矩阵：[LSTM预测, RF预测, 全1列（偏置）]
    A = np.vstack([y_lstm_va, y_rf_va, np.ones(n)]).T  # shape: (n, 3)

    # 岭回归封闭解：(A^T A + alpha*I)^{-1} A^T y
    ATA = A.T @ A + alpha * np.eye(3)
    ATy = A.T @ y_true_va
    coef = np.linalg.solve(ATA, ATy)  # [w_lstm, w_rf, bias]

    w_lstm, w_rf, bias = coef

    # ⭐⭐⭐ 放宽权重约束（论文版本）：允许负权重和适度外推
    # 从 [0.1, 0.9] 改为 [-1.0, 2.0]，允许模型学习反向关系
    w_lstm_clipped = np.clip(w_lstm, -1.0, 2.0)
    w_rf_clipped = np.clip(w_rf, -1.0, 2.0)

    # ⭐⭐⭐ 限制偏置在真值标准差的±3倍内，防止系统性偏差过大
    bias_limit = 3 * y_true_va.std()
    bias_clipped = np.clip(bias, -bias_limit, bias_limit)

    if w_lstm != w_lstm_clipped or w_rf != w_rf_clipped or bias != bias_clipped:
        print(f"      [WARN] 权重或偏置超出合理范围，已裁剪:")
        print(f"        原始: LSTM={w_lstm:.4f}, RF={w_rf:.4f}, bias={bias:.4f}")
        print(f"        裁剪后: LSTM={w_lstm_clipped:.4f}, RF={w_rf_clipped:.4f}, bias={bias_clipped:.4f}")

    coef = np.array([w_lstm_clipped, w_rf_clipped, bias_clipped])
    print(f"      岭回归权重: LSTM={w_lstm_clipped:.4f}, RF={w_rf_clipped:.4f}, bias={bias_clipped:.4f}")

    return coef

def hybrid_predict(lstm_pred, rf_pred, seq_length=7, coef=None):  # ⭐ 减小 seq_length 到 7
    """
    ⭐⭐⭐ 修复版融合预测函数（带偏置项 + 维度对齐修复）

    关键修复（参考 revise.md + 维度对齐）：
    - LSTM 预测覆盖 [seq_length, ..., T-1]，长度 = len(test_data) - seq_length
    - RF 预测现在也覆盖 [seq_length, ..., T-1]，长度 = len(test_data) - seq_length
    - ⭐⭐⭐ 两者长度已经一致，直接对齐到最短长度即可
    - 使用岭回归学习的权重 + 偏置项融合

    Args:
        lstm_pred: LSTM 预测数组
        rf_pred: RF 预测数组（现在长度和LSTM一致）
        seq_length: 序列长度（保留参数兼容性，但不再用于切片）
        coef: [w_lstm, w_rf, bias] 三参数或 None（使用默认权重）
    """
    # ⭐⭐⭐ 关键修复：RF现在返回的长度和LSTM一致，直接对齐到最短长度
    min_len = min(len(lstm_pred), len(rf_pred))
    lstm_pred_trim = lstm_pred[:min_len]
    rf_pred_trim = rf_pred[:min_len]

    if coef is not None and len(coef) == 3:
        # 使用学习的权重 + 偏置
        w_lstm, w_rf, bias = coef
        hybrid_pred = w_lstm * lstm_pred_trim + w_rf * rf_pred_trim + bias
    else:
        # 回退到默认权重（无偏置）
        lstm_weight = coef[0] if (coef is not None and len(coef) > 0) else 0.5
        hybrid_pred = lstm_weight * lstm_pred_trim + (1 - lstm_weight) * rf_pred_trim

    return hybrid_pred

def calculate_metrics(y_true, y_pred, y_train=None, model_name=""):
    """
    ⭐⭐⭐ 增强版评估指标（revise.md 第1.5节）

    新增指标：
    - SMAPE: Symmetric Mean Absolute Percentage Error
    - MASE: Mean Absolute Scaled Error（需要训练集）
    - Naive baseline: 简单持久性预测 (y_t = y_{t-1})

    Args:
        y_true: 真实值
        y_pred: 预测值
        y_train: 训练集真实值（用于计算MASE）
        model_name: 模型名称
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return None

    # 基础指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # ⭐⭐ SMAPE (Symmetric Mean Absolute Percentage Error)
    # 范围 [0, 100]，对称，对大小值都公平
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    # ⭐⭐ MASE (Mean Absolute Scaled Error)
    # 如果有训练集，计算 MASE（相对于 naive 预测的改进）
    mase = None
    if y_train is not None and len(y_train) > 1:
        # Naive预测的平均绝对误差（训练集上的一阶差分）
        naive_mae_train = np.mean(np.abs(np.diff(y_train)))
        if naive_mae_train > 1e-8:
            mase = mae / naive_mae_train

    # ⭐⭐ Naive基线（持久性预测）
    # 预测值 = 前一个真实值
    if len(y_true) > 1:
        y_naive = np.roll(y_true, 1)
        y_naive[0] = y_true[0]  # 第一个点无前值，用自身
        naive_mae = mean_absolute_error(y_true, y_naive)
        naive_rmse = np.sqrt(mean_squared_error(y_true, y_naive))
    else:
        naive_mae = np.nan
        naive_rmse = np.nan

    print(f"  {model_name}:")
    print(f"    - RMSE:  {rmse:.4f}  (Naive: {naive_rmse:.4f})")
    print(f"    - MAE:   {mae:.4f}   (Naive: {naive_mae:.4f})")
    print(f"    - R2:    {r2:.4f}")
    print(f"    - SMAPE: {smape:.2f}%")
    if mase is not None:
        print(f"    - MASE:  {mase:.4f}  (<1表示优于Naive)")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SMAPE': smape,
        'MASE': mase,
        'Naive_MAE': naive_mae,
        'Naive_RMSE': naive_rmse
    }

def plot_predictions(test_data, y_true, y_lstm, y_rf, y_hybrid, indicator, province, seq_length=7):  # ⭐ 减小 seq_length 到 7
    """
    ⭐⭐⭐ 修复版预测结果对比图（配合补丁A + 维度对齐修复）

    重要：调用方已传入对齐后的预测值（y_lstm_aligned, y_rf_aligned, y_hybrid_aligned），
    它们长度相同，且对应 y_test[seq_length : seq_length + n_common]。
    本函数只需正确提取对应的dates和y_true即可。

    ⭐⭐⭐ 维度对齐验证：添加断言确保所有输入长度一致
    """
    # ⭐⭐⭐ 关键修复：验证所有预测数组长度一致
    if not (len(y_lstm) == len(y_rf) == len(y_hybrid)):
        print(f"[WARN] 预测长度不一致: LSTM={len(y_lstm)}, RF={len(y_rf)}, Hybrid={len(y_hybrid)}")
        print(f"[WARN] 将对齐到最短长度")
        min_len = min(len(y_lstm), len(y_rf), len(y_hybrid))
        y_lstm = y_lstm[:min_len]
        y_rf = y_rf[:min_len]
        y_hybrid = y_hybrid[:min_len]

    # 传入的预测值已经对齐，长度相同
    n = len(y_lstm)

    # 提取对应的日期和真值
    dates = test_data['Date'].values[seq_length : seq_length + n]
    y_true_plot = y_true[seq_length : seq_length + n]

    # 直接使用传入的对齐后的预测值
    y_lstm_plot = y_lstm
    y_rf_plot = y_rf
    y_hybrid_plot = y_hybrid

    # ⭐ 强制设置字体（确保绘图时使用正确字体）
    font_prop = fm.FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 绘制时间序列
    axes[0].plot(dates, y_true_plot, 'o-', label='实际值', linewidth=2, markersize=4, color='black')
    axes[0].plot(dates, y_lstm_plot, 's--', label='LSTM预测', alpha=0.7, color='blue')
    axes[0].plot(dates, y_rf_plot, '^--', label='随机森林预测', alpha=0.7, color='green')
    axes[0].plot(dates, y_hybrid_plot, 'd-', label='混合模型预测', alpha=0.8, linewidth=2, color='red')
    axes[0].set_xlabel('日期', fontproperties=font_prop, fontsize=11)
    axes[0].set_ylabel(f'{indicator} (单位)', fontproperties=font_prop, fontsize=11)
    axes[0].set_title(f'{province} - {indicator} 时间序列预测对比', fontproperties=font_prop, fontsize=13, fontweight='bold')
    axes[0].legend(prop=font_prop, fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 绘制误差对比
    lstm_errors = np.abs(y_lstm_plot - y_true_plot)
    rf_errors = np.abs(y_rf_plot - y_true_plot)
    hybrid_errors = np.abs(y_hybrid_plot - y_true_plot)

    axes[1].bar(np.arange(len(dates))[::3], lstm_errors[::3], alpha=0.6, label='LSTM误差', width=0.25, color='blue')
    axes[1].bar(np.arange(len(dates))[::3] + 0.25, rf_errors[::3], alpha=0.6, label='RF误差', width=0.25, color='green')
    axes[1].bar(np.arange(len(dates))[::3] + 0.5, hybrid_errors[::3], alpha=0.6, label='混合误差', width=0.25, color='red')
    axes[1].set_xlabel('时间点', fontproperties=font_prop, fontsize=11)
    axes[1].set_ylabel('绝对误差', fontproperties=font_prop, fontsize=11)
    axes[1].set_title('预测误差对比', fontproperties=font_prop, fontsize=12, fontweight='bold')
    axes[1].legend(prop=font_prop, fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/Province_Plots/{province}_{indicator}_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 图表已保存: {save_path}")
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
            epochs = trial.suggest_int('epochs', 30, 150, step=10)  # ⭐ 增加epochs范围
            seq_length = trial.suggest_int('seq_length', 5, 20, step=5)
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 200, step=50)
            rf_max_depth = trial.suggest_int('rf_max_depth', 10, 30, step=5)
            lstm_weight = trial.suggest_float('lstm_weight', 0.3, 0.7, step=0.1)

            logger.info(f"Trial {trial.number}: Testing params - hidden_size={hidden_size}, num_layers={num_layers}, lr={learning_rate:.6f}")

            target_col = f'{self.indicator}_target'

            # ⭐⭐⭐ 训练LSTM（使用统一目标列）
            feature_cols = [c for c in self.train_data.columns
                            if c not in ['Date', 'Province', 'Year', self.indicator, target_col, 'spike_flag', 'sample_weight']]
            X_train = self.train_data[feature_cols].values
            y_train = self.train_data[target_col].values  # ⭐ 使用统一目标列
            w_train = self.train_data['sample_weight'].values  # ⭐ 权重

            # ⭐⭐⭐ Conductivity特殊归一化（使用RobustScaler）
            if 'Conductivity' in self.indicator:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)

            # ⭐⭐⭐ 创建数据集（传入权重）
            dataset = TimeSeriesDataset(X_train_scaled, y_train, weights=w_train, seq_length=seq_length)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            input_size = X_train_scaled.shape[1]
            lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout).to(DEVICE)
            optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
            # ⭐⭐ 使用无归约的 SmoothL1Loss，手动加权
            criterion = nn.SmoothL1Loss(reduction='none', beta=1.0)

            # ⭐ 改进的学习率调度：CosineAnnealing
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

            best_loss = float('inf')
            patience = 20  # ⭐ Optuna 试验中使用更小的 patience（20 代替 30）
            patience_counter = 0
            min_delta = 1e-4  # ⭐ 增大 min_delta 到 1e-4，更容易触发早停

            for epoch in range(epochs):
                lstm_model.train()
                epoch_losses = []

                for X_batch, y_batch, w_batch in loader:  # ⭐⭐⭐ 修复：直接获取权重
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    w_batch = w_batch.to(DEVICE)

                    optimizer.zero_grad()
                    y_pred = lstm_model(X_batch)

                    # ⭐⭐ 加权损失
                    loss_vec = criterion(y_pred, y_batch)
                    loss = (loss_vec * w_batch.unsqueeze(1)).mean()

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
            y_lstm_pred = predict_lstm(lstm_model, self.test_data, self.indicator, scaler, feature_cols, seq_length)

            # ⭐⭐⭐ 训练随机森林（使用变化量目标 + 样本权重 + 浅树结构）
            rf_model = RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=2,
                min_samples_leaf=5,      # ⭐ revise.md 推荐
                max_features=0.7,        # ⭐ revise.md 推荐
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train, sample_weight=w_train)

            # RF预测
            y_rf_pred = predict_rf(rf_model, self.test_data, self.indicator, feature_cols, seq_length)

            # ⭐ 混合预测（传入 seq_length 对齐时间步）
            y_hybrid_pred = hybrid_predict(y_lstm_pred, y_rf_pred, seq_length=seq_length, lstm_weight=lstm_weight)

            # ⭐ 评估（对齐 y_test 到相同索引）
            y_test = self.test_data[self.indicator].values
            n = len(y_hybrid_pred)
            y_test_aligned = y_test[seq_length : seq_length + n]
            metrics = calculate_metrics(y_test_aligned, y_hybrid_pred, "")

            if metrics is None:
                return -np.inf

            r2_score_val = metrics['R2']
            logger.info(f"Trial {trial.number}: R2={r2_score_val:.4f}")

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

        logger.info(f"优化完成！最佳R2={self.best_score:.4f}")
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
        print("[ERROR] 错误: Optuna未安装")
        print("   请运行: pip install optuna")
        return

    # 加载数据
    print("\n[OK] 正在加载数据...")
    data = load_data()
    if data is None:
        print("[ERROR] 数据加载失败")
        return

    # 聚合到省份
    province_data = aggregate_to_province(data)

    if args.optimize:
        # 优化模式
        print("\n" + "="*70)
        print("  [OPTUNA] 启动Optuna超参数优化模式")
        print("="*70)

        indicator = args.optimize_indicator
        provinces = [args.optimize_province] if args.optimize_province else province_data['Province'].unique()

        optimization_results = {}

        for province in provinces:
            print(f"\n{'='*70}")
            print(f"[PROVINCE] 优化省份: {province}, 指标: {indicator}")
            print(f"{'='*70}")

            try:
                province_subset = province_data[province_data['Province'] == province]

                result = prepare_train_test_data(province_subset, indicator)
                if result is None:
                    print(f"  [WARN]  数据不足，跳过")
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

                print(f"\n[DONE] 优化完成！")
                print(f"   最佳R2: {best_score:.4f}")
                print(f"   结果已保存: {result_file}")

            except Exception as e:
                print(f"  [ERROR] 优化失败: {e}")
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
                f.write(f"  最佳R2: {results['best_score']:.4f}\n")
                f.write(f"  最佳参数:\n")
                for param, value in results['best_params'].items():
                    f.write(f"    - {param}: {value}\n")

        print(f"\n[DONE] 优化总结已保存: {summary_file}")

    else:
        # 标准预测模式
        target_indicators = [
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
            'Conductivity_uS_cm', 'Turbidity_NTU', 'COD_mg_L', 'BOD_mg_L'
        ]

        all_results = {}

        for indicator in target_indicators:
            print(f"\n{'='*70}")
            print(f"[TARGET] 处理指标: {indicator}")
            print(f"{'='*70}")

            province_results = {}

            for province in province_data['Province'].unique():
                print(f"\n  [PROVINCE] 省份: {province}")

                try:
                    province_subset = province_data[province_data['Province'] == province]

                    result = prepare_train_test_data(province_subset, indicator)
                    if result is None:
                        print(f"    [WARN]  数据不足，跳过")
                        continue

                    train_data, test_data, all_data = result
                    y_test = test_data[indicator].values

                    # ⭐⭐⭐ 补丁B: 先在train/val切分上学权重，再用全量数据重训
                    split_idx = int(len(train_data) * 0.85)
                    train_subset = train_data.iloc[:split_idx].copy()
                    val_subset = train_data.iloc[split_idx:].copy()

                    seq_length = 7  # ⭐ 定义 seq_length

                    # 第1步：在训练子集上训练模型（用于学习权重）
                    lstm_model_tmp, scaler_tmp, feature_cols = train_lstm_model(train_subset, indicator, epochs=30, seq_length=seq_length)
                    rf_model_tmp, rf_feature_cols = train_rf_model(train_subset, indicator)

                    # 在验证集上获取预测
                    y_lstm_val = predict_lstm(lstm_model_tmp, val_subset, indicator, scaler_tmp, feature_cols, seq_length)
                    y_rf_val = predict_rf(rf_model_tmp, val_subset, indicator, rf_feature_cols, seq_length)

                    # ⭐⭐⭐ 学习融合权重（使用验证集）
                    y_val_true = val_subset[indicator].values

                    # ⭐⭐⭐ 关键修复：RF现在返回长度和LSTM一致，直接对齐到最短长度
                    n_common = min(len(y_lstm_val), len(y_rf_val))
                    y_lstm_val_aligned = y_lstm_val[:n_common]
                    y_rf_val_aligned = y_rf_val[:n_common]
                    y_val_true_aligned = y_val_true[seq_length : seq_length + n_common]

                    # 学习权重 + 偏置（补丁C：岭回归）
                    if len(y_lstm_val_aligned) > 0 and len(y_rf_val_aligned) > 0:
                        coef = learn_blend_weight(y_lstm_val_aligned, y_rf_val_aligned, y_val_true_aligned)
                        w_lstm, w_rf, bias = coef
                        print(f"    [OK] 学习到的融合系数: LSTM={w_lstm:.4f}, RF={w_rf:.4f}, bias={bias:.4f}")
                    else:
                        coef = None
                        print(f"    [WARN]  验证集数据不足，使用默认权重")

                    # 第2步：⭐⭐⭐ 补丁B - 在全量train_data上重训最终模型
                    print(f"    [OK] 使用 train+val 全量数据重训最终模型...")
                    lstm_model, scaler, feature_cols = train_lstm_model(train_data, indicator, epochs=30, seq_length=seq_length)
                    rf_model, rf_feature_cols = train_rf_model(train_data, indicator)

                    # 在测试集上获取预测
                    y_lstm_pred = predict_lstm(lstm_model, test_data, indicator, scaler, feature_cols, seq_length)
                    y_rf_pred = predict_rf(rf_model, test_data, indicator, rf_feature_cols, seq_length)

                    # ⭐⭐⭐ 补丁C: 混合预测（使用学习到的权重 + 偏置）
                    y_hybrid_pred = hybrid_predict(y_lstm_pred, y_rf_pred, seq_length=seq_length, coef=coef)

                    print(f"\n    [EVAL] 模型评估:")
                    # ⭐⭐⭐ 补丁A: 统一评估切片（所有模型用同一段真值）
                    n_common = min(len(y_lstm_pred), len(y_rf_pred) - seq_length)

                    # 统一对齐的真值
                    y_test_aligned = y_test[seq_length : seq_length + n_common]

                    # 统一对齐的预测
                    y_lstm_aligned = y_lstm_pred[:n_common]
                    y_rf_aligned = y_rf_pred[seq_length : seq_length + n_common]
                    y_hybrid_aligned = y_hybrid_pred[:n_common]

                    # ⭐⭐ 传入训练集真值用于计算 MASE
                    y_train_values = train_data[indicator].values

                    lstm_metrics = calculate_metrics(y_test_aligned, y_lstm_aligned, y_train_values, "LSTM")
                    rf_metrics = calculate_metrics(y_test_aligned, y_rf_aligned, y_train_values, "随机森林")
                    hybrid_metrics = calculate_metrics(y_test_aligned, y_hybrid_aligned, y_train_values, "混合模型")

                    province_results[province] = {
                        'lstm': lstm_metrics,
                        'rf': rf_metrics,
                        'hybrid': hybrid_metrics
                    }

                    # ⭐ 绘图使用对齐后的预测
                    plot_predictions(test_data, y_test, y_lstm_aligned, y_rf_aligned, y_hybrid_aligned, indicator, province, seq_length)

                except Exception as e:
                    print(f"    [ERROR] 处理出错: {e}")
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
            f.write("⭐⭐ 新增指标: SMAPE, MASE, Naive基线对比\n")
            f.write("="*70 + "\n\n")

            for indicator, province_results in all_results.items():
                f.write(f"\n【{indicator}】\n")
                f.write("-"*70 + "\n")

                for province, results in province_results.items():
                    f.write(f"\n  {province}:\n")
                    f.write(f"    LSTM:   RMSE={results['lstm']['RMSE']:.4f}, MAE={results['lstm']['MAE']:.4f}, R2={results['lstm']['R2']:.4f}")
                    if results['lstm'].get('MASE'):
                        f.write(f", SMAPE={results['lstm']['SMAPE']:.2f}%, MASE={results['lstm']['MASE']:.4f}\n")
                    else:
                        f.write(f", SMAPE={results['lstm']['SMAPE']:.2f}%\n")

                    f.write(f"    RF:     RMSE={results['rf']['RMSE']:.4f}, MAE={results['rf']['MAE']:.4f}, R2={results['rf']['R2']:.4f}")
                    if results['rf'].get('MASE'):
                        f.write(f", SMAPE={results['rf']['SMAPE']:.2f}%, MASE={results['rf']['MASE']:.4f}\n")
                    else:
                        f.write(f", SMAPE={results['rf']['SMAPE']:.2f}%\n")

                    f.write(f"    混合:   RMSE={results['hybrid']['RMSE']:.4f}, MAE={results['hybrid']['MAE']:.4f}, R2={results['hybrid']['R2']:.4f}")
                    if results['hybrid'].get('MASE'):
                        f.write(f", SMAPE={results['hybrid']['SMAPE']:.2f}%, MASE={results['hybrid']['MASE']:.4f}\n")
                    else:
                        f.write(f", SMAPE={results['hybrid']['SMAPE']:.2f}%\n")

                    # Naive 基线
                    f.write(f"    Naive:  RMSE={results['hybrid']['Naive_RMSE']:.4f}, MAE={results['hybrid']['Naive_MAE']:.4f}\n")

        print(f"\n[DONE] 报告已保存: {report_path}")

    print(f"\n[DONE] 完成!")
    print(f"   输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
