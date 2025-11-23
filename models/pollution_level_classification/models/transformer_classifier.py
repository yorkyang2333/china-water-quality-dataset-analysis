#!/usr/bin/env python3
"""
Transformer水质分类器
使用自注意力机制处理水质特征，捕捉特征间的复杂关系

核心创新:
1. Multi-Head Self-Attention - 捕捉特征间的全局依赖关系
2. Positional Encoding - 为特征添加位置信息
3. Feed-Forward Network - 深度非线性变换
4. Layer Normalization - 稳定训练过程
5. Residual Connections - 缓解梯度消失
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import zhplot  # 自动配置matplotlib中文字体
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# 导入核心模块
try:
    from core.base_classifier import DeepLearningClassifier
    from core.safe_data_loader import WaterQualityDataLoader
    from core.evaluator import ClassifierEvaluator
except ImportError as e:
    print(f"[ERROR] 无法导入核心模块: {e}")
    print("[INFO] 请确保在 models/pollution_level_classification/ 目录下运行")
    sys.exit(1)


# ============================================================================
# Transformer核心组件
# ============================================================================

class PositionalEncoding(nn.Module):
    """位置编码 - 为每个特征添加位置信息"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask (optional)
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = x.size(0)

        # 线性投影并分成多头
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch_size, num_heads, seq_len, d_k)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)
        # output: (batch_size, num_heads, seq_len, d_k)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_o(output)

        return output, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask (optional)
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """Transformer编码器（多层堆叠）"""
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask (optional)
        Returns:
            output: (batch_size, seq_len, d_model)
            all_attention_weights: list of attention weights from each layer
        """
        all_attention_weights = []

        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)

        return x, all_attention_weights


# ============================================================================
# Transformer分类模型
# ============================================================================

class TransformerClassifier(nn.Module):
    """
    Transformer水质分类器

    架构:
    Input → Embedding → Positional Encoding → Transformer Encoder (N layers)
        → Global Average Pooling → Classifier Head → Output
    """
    def __init__(self, input_size, num_classes=5,
                 d_model=128, num_heads=8, num_layers=4, d_ff=512,
                 dropout=0.1, use_positional_encoding=True):
        super(TransformerClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # 输入嵌入（将每个特征映射到d_model维度）
        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 位置编码
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_len=input_size)

        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_size) - 原始特征
        Returns:
            output: (batch_size, num_classes) - 分类logits
            attention_weights: list of attention weights from each layer
        """
        batch_size = x.size(0)

        # 将输入reshape为序列格式: (batch_size, seq_len=input_size, 1)
        x = x.unsqueeze(-1)

        # 输入嵌入: (batch_size, seq_len, d_model)
        x = self.input_embedding(x)

        # 位置编码
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        # Transformer编码: (batch_size, seq_len, d_model)
        x, attention_weights = self.transformer_encoder(x)

        # 全局平均池化: (batch_size, d_model)
        x = x.mean(dim=1)

        # 分类: (batch_size, num_classes)
        output = self.classifier(x)

        return output, attention_weights


# ============================================================================
# PyTorch Dataset
# ============================================================================

class WaterQualityDataset(Dataset):
    """PyTorch数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# Transformer分类器包装类
# ============================================================================

class TransformerWaterQualityClassifier(DeepLearningClassifier):
    """Transformer水质分类器包装类"""

    def __init__(self, d_model=128, num_heads=8, num_layers=4, d_ff=512,
                 dropout=0.1, learning_rate=0.001, batch_size=64,
                 epochs=100, patience=15, use_positional_encoding=True):
        super().__init__(model_name='Transformer')

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.use_positional_encoding = use_positional_encoding

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Transformer] 使用设备: {self.device}")

    def build_model(self, input_size, num_classes=5):
        """构建Transformer模型"""
        print(f"[Transformer] 构建模型 - 输入维度: {input_size}, 类别数: {num_classes}")
        print(f"[Transformer] 超参数: d_model={self.d_model}, num_heads={self.num_heads}, "
              f"num_layers={self.num_layers}, d_ff={self.d_ff}")

        self.model = TransformerClassifier(
            input_size=input_size,
            num_classes=num_classes,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            use_positional_encoding=self.use_positional_encoding
        ).to(self.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Transformer] 总参数量: {total_params:,}")
        print(f"[Transformer] 可训练参数量: {trainable_params:,}")

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练模型"""
        print(f"\n[Transformer] 开始训练...")
        print(f"[Transformer] 训练集大小: {len(X_train)}")
        if X_val is not None:
            print(f"[Transformer] 验证集大小: {len(X_val)}")

        # 构建模型
        if self.model is None:
            input_size = X_train.shape[1]
            num_classes = len(np.unique(y_train))
            self.build_model(input_size, num_classes)

        # 创建数据加载器
        train_dataset = WaterQualityDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = WaterQualityDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        # 优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"[Transformer] Epoch [{epoch+1}/{self.epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                          f"Patience: {patience_counter}/{self.patience}")

                if patience_counter >= self.patience:
                    print(f"[Transformer] 早停触发 (Epoch {epoch+1})")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"[Transformer] Epoch [{epoch+1}/{self.epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            scheduler.step()

        # 恢复最佳模型
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f"[Transformer] 恢复最佳模型 (Val Loss: {best_val_loss:.4f})")

        self.is_trained = True
        self.training_history = history
        print(f"[Transformer] 训练完成！")

    def predict(self, X):
        """预测"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")

        self.model.eval()
        dataset = WaterQualityDataset(X, np.zeros(len(X)))  # dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs, _ = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        """预测概率"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")

        self.model.eval()
        dataset = WaterQualityDataset(X, np.zeros(len(X)))  # dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probas = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs, _ = self.model(batch_X)
                proba = F.softmax(outputs, dim=1)
                probas.extend(proba.cpu().numpy())

        return np.array(probas)

    def get_attention_weights(self, X, sample_idx=0):
        """获取注意力权重（用于可视化）"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X[sample_idx:sample_idx+1]).to(self.device)
            _, attention_weights = self.model(x)

        # 返回所有层的注意力权重
        return [attn.cpu().numpy() for attn in attention_weights]


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 80)
    print("Transformer水质分类器")
    print("=" * 80)

    # 加载数据
    print("\n[STEP 1] 加载数据...")
    loader = WaterQualityDataLoader(test_size=0.2, random_state=42)
    df = loader.load_raw_data()
    X_train, X_test, y_train, y_test, label_mapping = loader.preprocess(
        balance_method='smote',
        apply_smote=True
    )

    print(f"[OK] 训练集大小: {X_train.shape}")
    print(f"[OK] 测试集大小: {X_test.shape}")
    print(f"[OK] 类别映射: {label_mapping}")

    # 创建Transformer模型
    print("\n[STEP 2] 创建Transformer模型...")
    model = TransformerWaterQualityClassifier(
        d_model=128,           # 隐藏维度
        num_heads=8,           # 注意力头数
        num_layers=4,          # Transformer层数
        d_ff=512,              # 前馈网络维度
        dropout=0.1,           # Dropout率
        learning_rate=0.001,   # 学习率
        batch_size=64,         # 批大小
        epochs=150,            # 最大训练轮数
        patience=20,           # 早停patience
        use_positional_encoding=True  # 使用位置编码
    )

    # 训练模型
    print("\n[STEP 3] 训练模型...")
    # 划分训练/验证集
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    model.train(X_train_split, y_train_split, X_val_split, y_val_split)

    # 评估模型
    print("\n[STEP 4] 评估模型...")
    evaluator = ModelEvaluator(
        model=model,
        model_name='Transformer',
        output_dir='Outputs/Transformer_Classification'
    )

    # 生成评估报告
    evaluator.evaluate(X_test, y_test, label_mapping)
    evaluator.save_evaluation_report()
    evaluator.plot_confusion_matrix(X_test, y_test, label_mapping)

    # 保存模型
    print("\n[STEP 5] 保存模型...")
    model_path = os.path.join('Outputs/Transformer_Classification', 'transformer_model.pth')
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': {
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'num_layers': model.num_layers,
            'd_ff': model.d_ff,
            'dropout': model.dropout,
            'use_positional_encoding': model.use_positional_encoding
        },
        'label_mapping': label_mapping
    }, model_path)
    print(f"[OK] 模型已保存到: {model_path}")

    print("\n" + "=" * 80)
    print("Transformer分类器训练完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
