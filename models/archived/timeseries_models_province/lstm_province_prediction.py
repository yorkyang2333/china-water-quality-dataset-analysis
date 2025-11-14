"""
LSTM时间序列预测 - 按省份建模 (PyTorch版)

⚠️ 警告：此模型不适合当前数据集！
原因：数据时间间隔极不规则（0-222天），违反LSTM等间隔采样假设
结果：R² < 0，预测效果极差
建议：使用横截面预测或传统机器学习方法（详见 README.md）

方法：
- 使用PyTorch实现的3层LSTM神经网络（64 hidden units）
- 按省份聚合所有监测站数据
- 30个时间步的序列预测
- 数据归一化 + 早停策略 + 梯度裁剪

适用机型：
- Mac (推荐) - 优化了CPU性能
- Windows/Linux (CPU模式) - 通用兼容
- 无需GPU，适合笔记本电脑运行

性能：
- 训练速度：中等（CPU单线程）
- 内存需求：低（< 2GB）
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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/LSTM_Province_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""
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
    """PyTorch LSTM模型"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, dropout=0.3):
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
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm(model, train_loader, criterion, optimizer, device, epochs=150):
    """训练LSTM模型"""
    model.train()
    best_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def build_timeseries_lstm(province_data, indicator, seq_length=30, train_ratio=0.8):
    """
    使用PyTorch LSTM进行时间序列预测

    参数:
    - province_data: 省份数据（聚合所有监测站）
    - indicator: 预测指标
    - seq_length: 序列长度
    - train_ratio: 训练集比例
    """
    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    # 按时间排序
    data = data.sort_values('Date').reset_index(drop=True)

    # 只保留目标指标和日期
    data = data[['Date', indicator]].dropna()

    if len(data) < seq_length + 30:
        return None

    # 提取目标值
    values = data[indicator].values.reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    # 创建序列
    X, y = create_sequences(scaled_values, seq_length)

    if len(X) < 30:
        return None

    # 时间顺序分割
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) < 20 or len(X_test) < 5:
        return None

    try:
        # 设备配置
        device = torch.device('cpu')

        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        # 构建模型
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=3, dropout=0.3)
        model = model.to(device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

        # 训练模型
        model = train_lstm(model, train_loader, criterion, optimizer, device, epochs=150)

        # 预测
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()

        # 反归一化
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_original = scaler.inverse_transform(y_test).flatten()

        metrics = calculate_metrics(y_test_original, y_pred)
        if metrics is None:
            return None

        return {
            'model_name': 'LSTM',
            'y_test': y_test_original,
            'y_pred': y_pred,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metrics': metrics,
            'seq_length': seq_length
        }

    except Exception as e:
        return None


def predict_province(df, province, seq_length=30):
    """为单个省份预测所有指标（聚合所有监测站数据）"""
    province_data = df[df['Province'] == province].copy()

    if len(province_data) < seq_length + 30:
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

    for indicator in indicators:
        # 检查数据量
        valid_count = province_data[indicator].notna().sum()
        if valid_count < seq_length + 25:
            continue

        result = build_timeseries_lstm(province_data, indicator, seq_length)
        if result:
            results[indicator] = result

    return results if results else None


def process_all_provinces(df, seq_length=30):
    """处理所有省份"""
    print("\n" + "="*80)
    print("LSTM时间序列预测 - 按省份建模 (PyTorch版)")
    print("="*80)

    # 获取所有省份
    provinces = df['Province'].unique()
    province_counts = df.groupby('Province').size().reset_index(name='count')

    print(f"\n总省份数: {len(provinces)}")
    print(f"\n方法参数:")
    print(f"  - 序列长度: {seq_length}")
    print(f"  - 模型: 3层LSTM (64 units)")
    print(f"  - Epochs: 150 (patience=15)")
    print(f"  - 学习率: 0.0005")
    print(f"  - Batch size: 32")
    print(f"  - 建模方式: 按省份聚合所有监测站数据")

    all_results = {}
    successful_provinces = 0

    for idx, province in enumerate(sorted(provinces)):
        count = province_counts[province_counts['Province'] == province]['count'].values[0]

        print(f"\n[{idx+1}/{len(provinces)}] {province}")
        print(f"  总数据点数: {count}")

        results = predict_province(df, province, seq_length)

        if results:
            all_results[province] = {
                'province': province,
                'sample_count': count,
                'results': results
            }
            successful_provinces += 1

            # 显示结果摘要
            r2_scores = [r['metrics']['R2'] for r in results.values()]
            print(f"  ✓ 成功预测 {len(results)} 个指标")
            print(f"  平均 R²: {np.mean(r2_scores):.3f} (范围: {np.min(r2_scores):.3f} - {np.max(r2_scores):.3f})")
        else:
            print(f"  ✗ 数据不足，跳过")

    print(f"\n成功建模的省份: {successful_provinces}/{len(provinces)}")

    return all_results


def visualize_province(province, province_info, output_dir):
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

        y_test = result['y_test']
        y_pred = result['y_pred']
        m = result['metrics']

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#3498db')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        textstr = f"Model: LSTM\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{province} Province\nLSTM Time Series Prediction (Province-Level Aggregation)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{province}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_results, output_dir):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LSTM时间序列预测报告 - 按省份建模 (PyTorch版)\n")
        f.write("LSTM Time Series Prediction Report - Province-Based Modeling\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个省份独立建模（聚合省内所有监测站数据）\n")
        f.write("- 使用PyTorch实现的3层LSTM神经网络（64 hidden units）\n")
        f.write("- 序列长度: 30个时间步\n")
        f.write("- 数据归一化: MinMaxScaler\n")
        f.write("- 早停策略: patience=15\n")
        f.write("- 梯度裁剪: max_norm=1.0\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []

        for province, province_info in all_results.items():
            for indicator, result in province_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"省份数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n\n")

        f.write(f"R² 统计:\n")
        f.write(f"  平均值: {np.mean(all_r2):.4f}\n")
        f.write(f"  中位数: {np.median(all_r2):.4f}\n")
        f.write(f"  标准差: {np.std(all_r2):.4f}\n")
        f.write(f"  最大值: {np.max(all_r2):.4f}\n")
        f.write(f"  最小值: {np.min(all_r2):.4f}\n\n")

        # R²分布
        r2_ranges = {
            'R² > 0.7 (优秀)': sum(1 for r in all_r2 if r > 0.7),
            'R² 0.5-0.7 (良好)': sum(1 for r in all_r2 if 0.5 <= r <= 0.7),
            'R² 0.3-0.5 (中等)': sum(1 for r in all_r2 if 0.3 <= r < 0.5),
            'R² < 0.3 (较差)': sum(1 for r in all_r2 if r < 0.3)
        }

        f.write(f"R² 分布:\n")
        for range_name, count in r2_ranges.items():
            percentage = count / len(all_r2) * 100 if len(all_r2) > 0 else 0
            f.write(f"  {range_name}: {count} ({percentage:.1f}%)\n")
        f.write("\n")

        f.write(f"RMSE 平均值: {np.mean(all_rmse):.4f}\n")
        f.write(f"MAE 平均值: {np.mean(all_mae):.4f}\n\n")

        f.write("\n" + "="*80 + "\n\n")

        # 详细的省份结果 - 按R²排序
        f.write("详细结果 - 按R²排序\n")
        f.write("="*80 + "\n\n")

        # 收集所有省份的平均R²
        province_avg_r2 = []
        for province, province_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in province_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            province_avg_r2.append((province, province_info, avg_r2))

        # 按平均R²排序
        province_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for province, province_info, avg_r2 in province_avg_r2:
            f.write(f"省份: {province}\n")
            f.write(f"样本数: {province_info['sample_count']} | 平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'模型':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            # 按R²排序
            sorted_results = sorted(province_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                f.write(f"{indicator:<30} {'LSTM':<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个省份独立建立LSTM深度学习模型（聚合省内所有监测站）\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- LSTM能够捕捉复杂的非线性时间序列模式\n")
        f.write("- 使用PyTorch框架，优化了CPU性能\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("LSTM时间序列预测系统 - 按省份建模 (PyTorch版)")
    print("LSTM Time Series Prediction - Province-Based Modeling")
    print("="*80)

    # 加载数据
    print("\n[1/4] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ 总记录数: {len(df)}")
    print(f"✓ 省份数: {df['Province'].nunique()}")
    print(f"✓ 监测站数: {df['Monitoring_Station'].nunique()}")

    # 处理所有省份
    print("\n[2/4] 按省份建立LSTM模型...")
    all_results = process_all_provinces(df, seq_length=30)

    # 生成可视化
    print("\n[3/4] 生成可视化...")
    total = len(all_results)
    for idx, (province, province_info) in enumerate(all_results.items(), 1):
        print(f"  进度: {idx}/{total}")
        visualize_province(province, province_info, OUTPUT_DIR)

    # 生成报告
    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR)

    # 总结
    print("\n" + "="*80)
    print("完成！")
    print("="*80)

    total_models = sum(len(p['results']) for p in all_results.values())
    print(f"\n✓ 成功建模的省份: {len(all_results)}")
    print(f"✓ 总预测模型数: {total_models}")
    print(f"✓ 结果保存在: {OUTPUT_DIR}/")
    print(f"  - 报告: report.txt")
    print(f"  - 图表: {len(all_results)} 个省份的可视化\n")


if __name__ == "__main__":
    main()
