"""
LSTM时间序列预测 - ⭐⭐⭐ 强化版 + Optuna自动调参 + 模型保存 (Province级别)
LSTM Time Series Prediction - Enhanced with Auto Hyperparameter Tuning (Province)

⭐⭐⭐ 强化改进（v3.0 - 解决直线预测问题）：

🔥 模型架构强化：
1. ✅ Xavier/Orthogonal权重初始化 → 防止梯度消失
2. ✅ Batch Normalization → 提高训练稳定性
3. ✅ 3层全连接网络（原2层）→ 更强的表达能力
4. ✅ L2正则化 (weight_decay=1e-5) → 防止过拟合

🔥 训练策略强化：
5. ✅ epochs=500（原300）→ 更充分的训练
6. ✅ patience=100（原50）→ 避免过早停止
7. ✅ OneCycleLR调度器（原CosineAnnealing）→ 更好的学习率策略
8. ✅ Warmup learning rate（前30%时间）→ 训练更稳定
9. ✅ min_delta=1e-6（原1e-5）→ 更敏感的早停检测

🔥 超参数搜索强化：
10. ✅ Optuna trials: 50（原20）→ 更充分的搜索
11. ✅ hidden_size: 64-512（原64-256）→ 更大的搜索空间
12. ✅ num_layers: 2-5（原2-4）
13. ✅ dropout: 0.05-0.5（原0.1-0.4）
14. ✅ lr: 1e-5 to 5e-2（原1e-4 to 1e-2）
15. ✅ batch_size: 8-64（原8-32）

🔥 数据处理强化：
16. ✅ SEQUENCE_LENGTH: [5,10,15,20,25,30,40,50]（原[10,15,20,25,30]）
17. ✅ 最小数据要求: 60（原40）
18. ✅ 搜索SEQUENCE_LENGTH时epochs: 100（原50）

🔥 可视化特性：
19. ✅ 每完成一个特征立即生成图表（实时可视化）
20. ✅ 详细显示所有超参数信息

⚠️ 预期效果：
- 训练时间增加约2-3倍（值得！）
- 预测曲线不再是直线，能捕捉时间序列波动
- R²值显著提升
- 更好的泛化能力

依赖库：
pip install torch pandas numpy scikit-learn matplotlib optuna

日期：2025-10-29
版本：v3.0 (Enhanced - Solve Flat Prediction Problem)
"""

import zhplot  # 自动配置matplotlib中文字体
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
from optuna.trial import TrialState

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

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

OUTPUT_DIR = 'Outputs/LSTM_2023Only_Province_GPU_Optimized'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Province_Plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Saved_Models", exist_ok=True)

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
    ⭐⭐⭐ 强化版LSTM模型

    改进：
    - Xavier初始化防止梯度消失
    - Batch Normalization提高稳定性
    - 更深的全连接层
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

        # ⭐⭐ 添加Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        # ⭐⭐ 更深的全连接网络
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout * 0.5)  # 最后一层dropout减半

        self.fc3 = nn.Linear(hidden_size // 2, 1)

        # ⭐⭐ Xavier初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化，防止梯度消失"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步

        # ⭐⭐ Batch Normalization
        if out.size(0) > 1:  # 只有batch_size>1才能用BN
            out = self.batch_norm(out)

        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout3(out)
        out = self.fc3(out)
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
                     lr=0.001, epochs=500, batch_size=16, patience=100):
    """
    ⭐⭐⭐ 强化版训练函数

    改进：
    - epochs=500（原300）→ 更充分的训练
    - patience=100（原50）→ 避免过早停止
    - Warmup learning rate → 前10个epoch线性增加学习率
    - OneCycleLR scheduler → 更好的学习率策略
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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # ⭐ 添加L2正则化

    # ⭐⭐⭐ OneCycleLR: 更强大的学习率调度策略
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 前30%时间用于warmup
        anneal_strategy='cos'
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    min_delta = 1e-6  # ⭐ 更敏感的早停检测

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
            scheduler.step()  # ⭐ 每个batch后更新学习率
            train_losses.append(loss.item())

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

        # ⭐⭐ 改进的早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      早停于 Epoch {epoch+1}/{epochs}")
                break

        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

def objective(trial, X_train, y_train, X_val, y_val, device):
    """
    ⭐⭐⭐ Optuna目标函数（强化版）

    搜索空间扩大：
    - hidden_size: 64-512（原64-256）
    - num_layers: 2-5（原2-4）
    - dropout: 0.05-0.5（原0.1-0.4）
    - lr: 1e-5 to 5e-2（原1e-4 to 1e-2）
    - batch_size: 8-64（原8-32）
    """
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.05, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 5e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 64, step=8)

    model, history = train_lstm_model(
        X_train, y_train, X_val, y_val, device,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=200,  # 调参时用较少epochs（原150）
        batch_size=batch_size,
        patience=50
    )

    # 返回验证集最佳loss
    return min(history['val_loss'])

def predict_indicator_with_tuning(province_data, indicator, device, province_name, train_ratio=0.8):
    """
    ⭐⭐⭐ 强化版带Optuna自动调参的预测函数

    改进：
    - SEQUENCE_LENGTH搜索范围: [5,10,15,20,25,30,40,50]（原[10,15,20,25,30]）
    - Optuna trials: 50（原20）
    - 最终训练epochs: 500（原300）
    - 最终训练patience: 100（原50）
    """
    province_data = province_data.copy()
    province_data['Date'] = pd.to_datetime(province_data['Date'])
    province_data = province_data.sort_values('Date').reset_index(drop=True)

    province_data['Year'] = province_data['Date'].dt.year
    data_2023 = province_data[province_data['Year'] == 2023].copy()

    indicator_data = data_2023[indicator].dropna()
    if len(indicator_data) < 60:  # ⭐ 提高最小数据要求（原40）
        return None

    indicator_values = indicator_data.values
    dates = data_2023[data_2023[indicator].notna()]['Date'].values

    scaler = MinMaxScaler()
    indicator_scaled = scaler.fit_transform(indicator_values.reshape(-1, 1)).flatten()

    # ⭐⭐⭐ Step 1: 搜索最佳SEQUENCE_LENGTH（扩大搜索范围）
    best_seq_length = None
    best_seq_score = float('inf')

    print(f"      [1/3] 搜索最佳SEQUENCE_LENGTH（范围更广）...", end=" ")
    for seq_len in [5, 10, 15, 20, 25, 30, 40, 50]:  # ⭐ 增加候选值
        X, y = create_sequences(indicator_scaled, seq_len)
        if len(X) < 30:  # ⭐ 提高最小序列要求（原20）
            continue

        split_idx = int(len(X) * train_ratio)
        if split_idx < 15:  # ⭐ 提高训练集最小要求（原10）
            continue

        X_train_all, X_test = X[:split_idx], X[split_idx:]
        y_train_all, y_test = y[:split_idx], y[split_idx:]

        val_split = int(len(X_train_all) * 0.8)
        X_train, X_val = X_train_all[:val_split], X_train_all[val_split:]
        y_train, y_val = y_train_all[:val_split], y_train_all[val_split:]

        if len(X_val) < 3:  # ⭐ 提高验证集最小要求（原2）
            continue

        # ⭐ 快速测试时用更多epochs（原50→100）
        model, history = train_lstm_model(
            X_train, y_train, X_val, y_val, device,
            hidden_size=128, num_layers=2, dropout=0.2,
            lr=0.001, epochs=100, batch_size=16, patience=20
        )

        val_loss = min(history['val_loss'])
        if val_loss < best_seq_score:
            best_seq_score = val_loss
            best_seq_length = seq_len

    if best_seq_length is None:
        print("失败")
        return None

    print(f"最佳={best_seq_length}")

    # 使用最佳SEQUENCE_LENGTH
    X, y = create_sequences(indicator_scaled, best_seq_length)
    split_idx = int(len(X) * train_ratio)
    X_train_all, X_test = X[:split_idx], X[split_idx:]
    y_train_all, y_test = y[:split_idx], y[split_idx:]

    val_split = int(len(X_train_all) * 0.8)
    X_train, X_val = X_train_all[:val_split], X_train_all[val_split:]
    y_train, y_val = y_train_all[:val_split], y_train_all[val_split:]

    # ⭐⭐⭐ Step 2: Optuna自动调参（增加trials）
    print(f"      [2/3] Optuna自动调参 (50 trials, 更广搜索空间)...")
    study = optuna.create_study(direction='minimize', study_name=f'{province_name}_{indicator}')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, device),
        n_trials=50,  # ⭐ 增加trials（原20）
        show_progress_bar=False
    )

    best_params = study.best_params
    print(f"      最佳参数: hidden={best_params['hidden_size']}, layers={best_params['num_layers']}, "
          f"lr={best_params['lr']:.5f}, dropout={best_params['dropout']:.3f}, batch={best_params['batch_size']}")

    # ⭐⭐⭐ Step 3: 使用最佳参数训练最终模型（更多epochs和patience）
    print(f"      [3/3] 训练最终模型（epochs=500, patience=100）...")
    final_model, history = train_lstm_model(
        X_train_all, y_train_all, X_val, y_val, device,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        lr=best_params['lr'],
        epochs=500,  # ⭐ 增加epochs（原300）
        batch_size=best_params['batch_size'],
        patience=100  # ⭐ 增加patience（原50）
    )

    # 保存模型
    model_filename = f"{OUTPUT_DIR}/Saved_Models/{province_name}_{indicator}.pth"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'best_params': best_params,
        'seq_length': best_seq_length,
        'scaler_min': scaler.data_min_[0],
        'scaler_max': scaler.data_max_[0]
    }, model_filename)

    # 预测
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_pred = final_model(X_test_tensor).squeeze().cpu().numpy()

    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    train_end_idx = split_idx + best_seq_length
    train_dates = dates[best_seq_length:train_end_idx]
    train_values = indicator_values[best_seq_length:train_end_idx]
    test_dates = dates[split_idx + best_seq_length:]

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

def visualize_single_indicator(province_name, indicator, result, output_dir):
    """
    ⭐⭐ NEW: 为单个指标立即生成可视化图表
    每完成一个指标的预测，就生成一幅独立的图
    """
    # 创建省份专属文件夹
    province_dir = f"{output_dir}/Province_Plots/{province_name}"
    os.makedirs(province_dir, exist_ok=True)

    # 创建单个图表
    plt.figure(figsize=(12, 6))

    y_train = result['y_train']
    train_dates = pd.to_datetime(result['train_dates'])
    y_test = result['y_test']
    y_pred = result['y_pred']
    test_dates = pd.to_datetime(result['test_dates'])
    m = result['metrics']

    # 训练数据（绿色）
    plt.plot(train_dates, y_train, marker='o', linewidth=2, markersize=4,
             color='green', label='训练数据 (80%)', alpha=0.7)

    # 测试数据实际值（蓝色）
    plt.plot(test_dates, y_test, marker='o', linewidth=2.5, markersize=6,
             color='steelblue', label='真实值 (测试集 20%)', alpha=0.9, zorder=3)

    # 预测值（红色虚线）
    plt.plot(test_dates, y_pred, marker='s', linewidth=2.5, markersize=6,
             color='orangered', label='预测值 (LSTM)', alpha=0.9, linestyle='--', zorder=2)

    # 显示指标信息
    params = result['best_params']
    textstr = (f"R² = {m['R2']:.4f}\n"
               f"RMSE = {m['RMSE']:.3f}\n"
               f"MAE = {m['MAE']:.3f}\n"
               f"Test Samples = {m['Sample_Size']}\n"
               f"\n⚙️ 最佳参数:\n"
               f"SeqLen = {result['seq_length']}\n"
               f"Hidden = {params['hidden_size']}\n"
               f"Layers = {params['num_layers']}\n"
               f"LR = {params['lr']:.5f}\n"
               f"Dropout = {params['dropout']:.2f}")

    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

    # 标题和标签
    plt.title(f"{province_name} Province - {indicator.replace('_', ' ')}\n"
              f"LSTM Time Series Prediction (⭐⭐ Optimized + Auto-Tuned)",
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Date (2023 only, 80/20 split)', fontsize=11, fontweight='bold')
    plt.ylabel('Value', fontsize=11, fontweight='bold')
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图片
    filename = f'{province_dir}/{indicator}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"      📊 图表已保存: {filename}")

def predict_province(province_data, province_name, device):
    """预测所有16个水质指标（Province级别）- ⭐⭐ 每完成一个指标立即生成图表"""
    province_data = province_data.copy()
    province_data['Date'] = pd.to_datetime(province_data['Date'])
    province_data = province_data[province_data['Date'].dt.year == 2023].copy()

    if len(province_data) < 40:
        print(f"  ⚠️ {province_name}: 2023年数据不足（<40条），跳过")
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
    print(f"  训练 {province_name} LSTM模型（⭐⭐ Optuna自动调参）...")

    for idx, indicator in enumerate(indicators, 1):
        print(f"    [{idx}/{len(indicators)}] {indicator}:")
        result = predict_indicator_with_tuning(province_data, indicator, device, province_name)
        if result:
            results[indicator] = result
            print(f"      ✓ R²={result['metrics']['R2']:.4f}, 模型已保存: {result['model_file']}")

            # ⭐⭐ NEW: 立即生成图表
            visualize_single_indicator(province_name, indicator, result, OUTPUT_DIR)

    if not results:
        print(f"  ⚠️ {province_name}: 所有指标均失败")
        return None

    return results

def process_all_provinces(df, device):
    """处理所有省份"""
    print("\n" + "="*80)
    print("LSTM时间序列预测 - ⭐⭐ Optuna自动调参优化版 (Province级别)")
    print("="*80)

    provinces = df['Province'].unique()
    print(f"\n总省份数: {len(provinces)}")
    print(f"设备: {device}")

    all_results = {}
    successful = 0

    for idx, province in enumerate(provinces, 1):
        print(f"\n[{idx}/{len(provinces)}] {province}")

        province_data = df[df["Province"] == province].copy()
        results = predict_province(province_data, province, device)

        if results:
            all_results[province] = {
                'province': province,
                'results': results
            }
            successful += 1
            r2_values = [r['metrics']['R2'] for r in results.values()]
            avg_r2 = np.mean(r2_values)
            print(f"  ✓ {len(results)} 指标成功，平均 R² = {avg_r2:.4f}")

    print(f"\n成功: {successful}/{len(provinces)}")
    return all_results

def visualize_province(province_key, province_info, output_dir):
    """可视化单个省份的预测结果"""
    results = province_info['results']
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

        ax.plot(train_dates, y_train, marker='o', linewidth=1.5, markersize=3,
                color='green', label='训练数据', alpha=0.6)
        ax.plot(test_dates, y_test, marker='o', linewidth=2, markersize=5,
                color='steelblue', label='真实值 (测试集)', alpha=0.8)
        ax.plot(test_dates, y_pred, marker='s', linewidth=2, markersize=5,
                color='orangered', label='预测值 (LSTM)', alpha=0.8, linestyle='--')

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

    title = f"{province_info['province']} Province\n" + \
            f"LSTM Prediction (⭐⭐ Optimized + Auto-Tuned)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/Province_Plots/{province_key}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(all_results, output_dir, device):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LSTM时间序列预测报告 - ⭐⭐⭐ 强化版（v3.0）\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"计算设备: {device}\n\n")

        f.write("⭐⭐⭐ 强化改进（v3.0 - 解决直线预测问题）：\n")
        f.write("-" * 80 + "\n")
        f.write("🔥 模型架构强化:\n")
        f.write("  ✅ Xavier/Orthogonal权重初始化 → 防止梯度消失\n")
        f.write("  ✅ Batch Normalization → 提高训练稳定性\n")
        f.write("  ✅ 3层全连接网络（更深）→ 更强的表达能力\n")
        f.write("  ✅ L2正则化 → 防止过拟合\n\n")

        f.write("🔥 训练策略强化:\n")
        f.write("  ✅ epochs=500（原300）→ 更充分的训练\n")
        f.write("  ✅ patience=100（原50）→ 避免过早停止\n")
        f.write("  ✅ OneCycleLR调度器 → 更好的学习率策略\n")
        f.write("  ✅ Warmup learning rate → 训练更稳定\n\n")

        f.write("🔥 超参数搜索强化:\n")
        f.write("  ✅ Optuna trials: 50（原20）→ 更充分的搜索\n")
        f.write("  ✅ 搜索空间大幅扩大（hidden: 64-512, layers: 2-5, etc.）\n")
        f.write("  ✅ SEQUENCE_LENGTH: [5,10,15,20,25,30,40,50]\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("✓ ⭐ 仅使用2023年的数据\n")
        f.write("✓ ⭐ 训练集：2023年数据的前80%（按时间顺序）\n")
        f.write("✓ ⭐ 测试集：2023年数据的后20%（按时间顺序）\n")
        f.write("✓ 聚合：按省份聚合所有监测站数据\n")
        f.write("✓ 模型：强化版LSTM + BatchNorm + 深层FC\n\n")

        all_r2, all_rmse, all_mae = [], [], []

        for province_info in all_results.values():
            for indicator, result in province_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"省份数量: {len(all_results)}\n")
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

        province_avg_r2 = []
        for province_key, province_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in province_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            province_avg_r2.append((province_key, province_info, avg_r2))

        province_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for province_key, province_info, avg_r2 in province_avg_r2:
            f.write(f"省份: {province_info['province']}\n")
            f.write(f"平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'SeqLen':>7} {'Hidden':>7} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            sorted_results = sorted(province_info['results'].items(),
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
        f.write("- 每个省份的每个指标独立使用Optuna自动调参\n")
        f.write("- SeqLen: 自动搜索的最佳SEQUENCE_LENGTH\n")
        f.write("- Hidden: 自动搜索的最佳隐藏层大小\n")
        f.write("- ⭐⭐ 所有最佳模型已保存至 Saved_Models/ 文件夹\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")

def main():
    """主函数"""
    print("="*80)
    print("LSTM时间序列预测 - ⭐⭐ Optuna自动调参优化版 (Province级别)")
    print("⭐⭐ NEW: 每完成一个特征立即生成图表")
    print("="*80)

    print("\n[1/3] 加载数据...")
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

    print("\n[2/3] 训练LSTM模型（⭐⭐ 每完成一个指标立即生成图表）...")
    all_results = process_all_provinces(df, DEVICE)

    if not all_results:
        print("\n⚠️ 没有符合条件的省份")
        return

    # ⭐⭐ NOTE: 可视化已在训练过程中实时生成，不需要额外步骤
    print("\n✓ 所有图表已在训练过程中实时生成")

    print("\n[3/3] 生成综合报告...")
    generate_report(all_results, OUTPUT_DIR, DEVICE)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n✓ 成功建模: {len(all_results)} 个省份")
    print(f"✓ 结果保存: {OUTPUT_DIR}/")
    print(f"✓ 图表保存: {OUTPUT_DIR}/Province_Plots/<省份名>/<指标名>.png")
    print(f"✓ 模型保存: {OUTPUT_DIR}/Saved_Models/\n")

if __name__ == "__main__":
    main()
