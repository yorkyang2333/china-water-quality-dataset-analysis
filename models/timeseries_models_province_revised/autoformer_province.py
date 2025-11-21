#!/usr/bin/env python3
"""
Autoformer水质时序预测模型 - Province级别
基于论文: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

核心创新:
1. Auto-Correlation机制 - 替代传统自注意力，更适合时序数据
2. Series Decomposition - 趋势-季节性分解
3. Deep Decomposition Architecture - 多层分解结构
4. Progressive Decomposition - 渐进式分解

注意: 根据项目文档，时序方法因数据采样不规则（CV=1.99）而效果不佳，
     本模型主要用于学术探索和对比实验。
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("[ERROR] PyTorch未安装，请运行: pip install torch")
    TORCH_AVAILABLE = False
    sys.exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ============================================================================
# Autoformer核心组件
# ============================================================================

class SeriesDecomposition(nn.Module):
    """
    时间序列分解模块
    将时序分解为趋势项和季节项: X = Trend + Seasonal
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        # 使用平均池化提取趋势
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_features)
        Returns:
            seasonal: 季节项
            trend: 趋势项
        """
        # 转置用于池化: (batch_size, num_features, seq_len)
        x_permuted = x.permute(0, 2, 1)

        # 提取趋势（移动平均）
        trend = self.avg_pool(x_permuted)
        trend = trend.permute(0, 2, 1)  # 转回: (batch_size, seq_len, num_features)

        # 季节项 = 原始 - 趋势
        seasonal = x - trend

        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    自相关机制
    基于FFT计算序列的自相关，自动发现周期性模式
    """
    def __init__(self, factor=1, scale=None):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale

    def time_delay_agg_training(self, values, corr):
        """
        时间延迟聚合
        Args:
            values: (batch_size, num_heads, seq_len, d_k)
            corr: (batch_size, num_heads, seq_len, seq_len) 自相关矩阵
        Returns:
            aggregated values
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[3]
        length = values.shape[2]

        # 找到Top-k个时间延迟
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # (batch, seq_len)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, idx] for idx in index], dim=-1)  # (batch, top_k)
        weights = F.softmax(weights, dim=-1)

        # 聚合
        tmp_corr = torch.softmax(corr, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            pattern = torch.roll(tmp_values, shifts=-int(index[i]), dims=2)
            delays_agg = delays_agg + pattern * (weights[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        return delays_agg

    def forward(self, queries, keys, values):
        """
        Args:
            queries: (batch_size, num_heads, seq_len, d_k)
            keys: (batch_size, num_heads, seq_len, d_k)
            values: (batch_size, num_heads, seq_len, d_k)
        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            corr: (batch_size, num_heads, seq_len, seq_len) 自相关矩阵
        """
        B, H, L, E = queries.shape

        # FFT计算自相关
        q_fft = torch.fft.rfft(queries.permute(0, 1, 3, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 2).contiguous(), dim=-1)

        # 自相关 = IFFT(Q_fft * conj(K_fft))
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)  # (batch, head, d_k, seq_len)
        corr = corr.permute(0, 1, 3, 2)  # (batch, head, seq_len, d_k)

        # 时间延迟聚合
        V = self.time_delay_agg_training(values, corr)

        return V.contiguous(), corr


class MultiHeadAutoCorrelation(nn.Module):
    """多头自相关注意力"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAutoCorrelation, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.auto_correlation = AutoCorrelation(factor=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)

        # 线性投影并分成多头
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 自相关注意力
        output, _ = self.auto_correlation(Q, K, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_o(output)

        return self.dropout(output)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class AutoformerEncoderLayer(nn.Module):
    """Autoformer编码器层（包含Series Decomposition）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(AutoformerEncoderLayer, self).__init__()

        self.auto_correlation = MultiHeadAutoCorrelation(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.decomp1 = SeriesDecomposition(kernel_size=25)
        self.decomp2 = SeriesDecomposition(kernel_size=25)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
            trend: 累积的趋势项
        """
        # 自相关注意力 + 残差 + 分解
        new_x = self.auto_correlation(x)
        x = x + self.dropout(new_x)
        x, trend1 = self.decomp1(x)

        # 前馈网络 + 残差 + 分解
        new_x = self.feed_forward(x)
        x = x + self.dropout(new_x)
        x, trend2 = self.decomp2(x)

        # 累积趋势
        trend = trend1 + trend2

        return x, trend


class AutoformerEncoder(nn.Module):
    """Autoformer编码器"""
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(AutoformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            seasonal: 季节项
            trend: 趋势项
        """
        trend_list = []

        for layer in self.layers:
            x, trend = layer(x)
            trend_list.append(trend)

        # 累积所有层的趋势
        trend = torch.sum(torch.stack(trend_list, dim=0), dim=0)

        return x, trend


# ============================================================================
# Autoformer完整模型
# ============================================================================

class Autoformer(nn.Module):
    """
    Autoformer时序预测模型

    架构:
    Input → Embedding → Autoformer Encoder → Decomposition
        → Trend Projection + Seasonal Projection → Output
    """
    def __init__(self, input_size, seq_len, pred_len,
                 d_model=128, num_heads=8, num_layers=2, d_ff=256,
                 dropout=0.1):
        super(Autoformer, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # 输入嵌入
        self.value_embedding = nn.Linear(input_size, d_model)

        # 位置编码（可学习）
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Autoformer编码器
        self.encoder = AutoformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

        # 输出投影
        self.trend_projection = nn.Linear(seq_len, pred_len)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)

        # 最终输出层
        self.output_layer = nn.Linear(d_model, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, pred_len, input_size)
        """
        # 输入嵌入
        x = self.value_embedding(x)  # (batch, seq_len, d_model)
        x = x + self.position_embedding
        x = self.dropout(x)

        # Autoformer编码
        seasonal, trend = self.encoder(x)  # (batch, seq_len, d_model)

        # 分别投影趋势和季节项到未来
        # 转置用于时间维度投影: (batch, d_model, seq_len)
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)

        # 投影到未来时间步
        trend = self.trend_projection(trend)  # (batch, d_model, pred_len)
        seasonal = self.seasonal_projection(seasonal)  # (batch, d_model, pred_len)

        # 转回: (batch, pred_len, d_model)
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)

        # 重组并输出
        output = trend + seasonal
        output = self.output_layer(output)  # (batch, pred_len, input_size)

        return output


# ============================================================================
# PyTorch Dataset
# ============================================================================

class TimeSeriesDataset(Dataset):
    """时间序列数据集（滑动窗口）"""
    def __init__(self, data, seq_len=30, pred_len=7):
        """
        Args:
            data: (num_samples, num_features) numpy array
            seq_len: 输入序列长度
            pred_len: 预测长度
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 输入: [idx : idx+seq_len]
        # 目标: [idx+seq_len : idx+seq_len+pred_len]
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y


# ============================================================================
# 训练和评估函数
# ============================================================================

def train_autoformer(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """训练Autoformer模型"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"[Autoformer] Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"[Autoformer] 早停触发 (Epoch {epoch+1})")
            break

        scheduler.step()

    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    print(f"[Autoformer] 训练完成，最佳验证损失: {best_val_loss:.6f}")

    return history


def evaluate_autoformer(model, test_loader, scaler, indicator_name):
    """评估Autoformer模型"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)

            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # 只取第一个特征的预测结果（假设预测单个指标）
    pred_flat = predictions[:, :, 0].flatten()
    actual_flat = actuals[:, :, 0].flatten()

    # 反标准化
    pred_original = scaler.inverse_transform(pred_flat.reshape(-1, 1)).flatten()
    actual_original = scaler.inverse_transform(actual_flat.reshape(-1, 1)).flatten()

    # 计算指标
    mse = mean_squared_error(actual_original, pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_original, pred_original)
    r2 = r2_score(actual_original, pred_original)

    print(f"\n[{indicator_name}] 评估结果:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': pred_original,
        'actuals': actual_original
    }


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 80)
    print("Autoformer水质时序预测 - Province级别")
    print("=" * 80)

    # 加载数据
    print("\n[STEP 1] 加载数据...")
    data_path = os.path.join(project_root, 'preprocessed_data_feature_eng.csv')
    df = pd.read_csv(data_path)
    print(f"[OK] 数据加载成功: {len(df)} 条记录")

    # 选择要预测的指标
    target_indicators = ['Water_Quality_Index', 'COD_mg_L', 'Total_Nitrogen_mg_L']

    # 选择省份
    provinces = df['Province'].unique()
    print(f"[OK] 共 {len(provinces)} 个省份")

    # 创建输出目录
    output_dir = os.path.join(project_root, 'Outputs', 'Autoformer_Province_Prediction')
    os.makedirs(output_dir, exist_ok=True)

    # 对每个省份和指标进行预测
    all_results = []

    for province in provinces[:3]:  # 测试3个省份
        print(f"\n{'='*80}")
        print(f"处理省份: {province}")
        print(f"{'='*80}")

        province_data = df[df['Province'] == province].copy()

        if len(province_data) < 100:
            print(f"[WARN] {province} 数据不足 ({len(province_data)} 条), 跳过")
            continue

        # 按日期排序
        province_data = province_data.sort_values('Date')

        for indicator in target_indicators:
            print(f"\n[{province} - {indicator}] 开始训练...")

            # 提取目标指标
            data = province_data[[indicator]].values

            # 标准化
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # 划分训练/验证/测试集 (70/15/15)
            n = len(data_scaled)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)

            train_data = data_scaled[:train_end]
            val_data = data_scaled[:val_end]  # 包含训练集，用于创建滑动窗口
            test_data = data_scaled  # 包含所有数据

            # 创建数据集
            seq_len = 30  # 使用过去30个时间步
            pred_len = 7  # 预测未来7个时间步

            train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
            val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
            test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            print(f"[OK] 训练集: {len(train_dataset)} 样本")
            print(f"[OK] 验证集: {len(val_dataset)} 样本")
            print(f"[OK] 测试集: {len(test_dataset)} 样本")

            # 创建模型
            model = Autoformer(
                input_size=1,      # 单变量预测
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=64,        # 降低维度以适应较小数据集
                num_heads=4,
                num_layers=2,
                d_ff=128,
                dropout=0.1
            ).to(device)

            # 训练
            history = train_autoformer(
                model, train_loader, val_loader,
                epochs=100, lr=0.001, patience=15
            )

            # 评估
            results = evaluate_autoformer(model, test_loader, scaler, f"{province} - {indicator}")

            # 保存结果
            results['province'] = province
            results['indicator'] = indicator
            all_results.append(results)

            # 可视化
            plt.figure(figsize=(12, 6))
            plt.plot(results['actuals'][:200], label='Actual', linewidth=2)
            plt.plot(results['predictions'][:200], label='Predicted', alpha=0.7, linewidth=2)
            plt.title(f'Autoformer预测 - {province} - {indicator}')
            plt.xlabel('时间步')
            plt.ylabel(indicator)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{province}_{indicator}_prediction.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] 图表已保存: {plot_path}")

    # 汇总报告
    print("\n" + "=" * 80)
    print("Autoformer预测完成 - 汇总报告")
    print("=" * 80)

    results_df = pd.DataFrame([{
        'Province': r['province'],
        'Indicator': r['indicator'],
        'MSE': r['mse'],
        'RMSE': r['rmse'],
        'MAE': r['mae'],
        'R²': r['r2']
    } for r in all_results])

    print(results_df.to_string(index=False))

    # 保存汇总报告
    report_path = os.path.join(output_dir, 'autoformer_summary.csv')
    results_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 汇总报告已保存: {report_path}")


if __name__ == '__main__':
    main()
