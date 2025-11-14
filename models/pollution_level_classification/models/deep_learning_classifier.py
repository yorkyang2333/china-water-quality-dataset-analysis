#!/usr/bin/env python3
"""
GPU加速深度学习水质分类模型
使用PyTorch实现多层感知机(MLP)和注意力机制

特点:
1. GPU加速训练 (如果可用)
2. 多层感知机架构
3. 自注意力机制
4. 处理类别不平衡的损失函数
5. 早停和学习率调度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# PyTorch相关
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print(f"🔥 PyTorch可用, GPU: {'是' if torch.cuda.is_available() else '否'}")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch未安装")

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WaterQualityDataset(Dataset):
    """PyTorch数据集类"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AttentionLayer(nn.Module):
    """自注意力层"""

    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, hidden_size)
        attention_weights = self.attention(x)  # (batch_size, 1)
        attention_weights = F.softmax(attention_weights, dim=0)
        return attention_weights

class WaterQualityNet(nn.Module):
    """深度学习水质分类网络"""

    def __init__(self, input_size, hidden_sizes=[128, 256, 128], num_classes=5, dropout_rate=0.3):
        super(WaterQualityNet, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # 构建隐藏层
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # 注意力层
        self.attention = AttentionLayer(hidden_sizes[-1])

        # 分类层
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 隐藏层前向传播
        hidden_output = self.hidden_layers(x)  # (batch_size, hidden_size)

        # 注意力机制
        attention_weights = self.attention(hidden_output)  # (batch_size, 1)
        attended_features = hidden_output * attention_weights  # (batch_size, hidden_size)

        # 分类输出
        output = self.classifier(attended_features)  # (batch_size, num_classes)

        return output, attention_weights

class DeepLearningClassifier:
    """深度学习分类器"""

    def __init__(self, output_dir='Outputs/DeepLearning_Classification'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def load_and_preprocess_data(self, csv_path):
        """加载和预处理数据"""
        print("🔄 加载和预处理数据...")

        df = pd.read_csv(csv_path)
        print(f"✅ 数据加载成功: {len(df)} 条记录")

        # 基础数据清理
        df['Date'] = pd.to_datetime(df['Date'])

        # 添加时间特征
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Date'].dt.month % 12 // 3 + 1
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

        # 周期性编码
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Season_sin'] = np.sin(2 * np.pi * df['Season'] / 4)
        df['Season_cos'] = np.cos(2 * np.pi * df['Season'] / 4)

        # 地理特征编码
        df['Province_encoded'] = df['Province'].astype('category').cat.codes
        df['City_encoded'] = df['City'].astype('category').cat.codes
        df['Station_encoded'] = df['Monitoring_Station'].astype('category').cat.codes

        # 空间几何特征
        df['Lat_Squared'] = df['Latitude'] ** 2
        df['Lon_Squared'] = df['Longitude'] ** 2
        df['Lat_Lon_Product'] = df['Latitude'] * df['Longitude']

        # 处理缺失值
        if 'NP_Ratio' in df.columns:
            df['NP_Ratio'].fillna(df['NP_Ratio'].median(), inplace=True)
        if 'COD_BOD_Ratio' in df.columns:
            df['COD_BOD_Ratio'].fillna(df['COD_BOD_Ratio'].median(), inplace=True)

        # 选择特征列
        feature_cols = [
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm',
            'Turbidity_NTU', 'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
            'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
            'Coliform_Count_CFU_100mL',
            'Total_Heavy_Metals_ug_L', 'COD_BOD_Ratio', 'NP_Ratio',
            'Latitude', 'Longitude', 'Province_encoded', 'City_encoded', 'Station_encoded',
            'Lat_Squared', 'Lon_Squared', 'Lat_Lon_Product',
            'Year', 'Month', 'Season', 'Quarter', 'DayOfWeek', 'IsWeekend',
            'Month_sin', 'Month_cos', 'Season_sin', 'Season_cos'
        ]

        # 移除不需要的列
        drop_cols = ['Province', 'City', 'Monitoring_Station', 'Date', 'Remarks', 'Water_Quality_Index']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # 确保所有特征列都存在
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"✅ 使用 {len(available_features)} 个特征")
        self.feature_names = available_features

        # 准备特征和目标
        X = df[available_features].copy()
        y = df['Pollution_Level'].copy()

        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        print(f"📊 标签映射: {self.label_mapping}")

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        print("\n🎯 类别分布:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            percentage = count / len(y_encoded) * 100
            print(f"   {label}: {count} 条 ({percentage:.1f}%)")

        return X_scaled, y_encoded

    def create_data_loaders(self, X, y, batch_size=64, test_size=0.2):
        """创建数据加载器"""

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 创建数据集
        train_dataset = WaterQualityDataset(X_train, y_train)
        test_dataset = WaterQualityDataset(X_test, y_test)

        # 计算类别权重用于采样 (使用温和的权重)
        class_counts = np.bincount(y_train)
        max_count = np.max(class_counts)
        # 使用更温和的权重：最大类别的权重为1，其他类别的权重为max_count/count
        class_weights = max_count / (class_counts + 1e-6)
        # 限制权重范围，避免过度加权
        class_weights = np.clip(class_weights, 0.5, 3.0)
        sample_weights = class_weights[y_train]

        # 创建加权采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, test_loader, X_test, y_test

    def create_model(self, input_size):
        """创建深度学习模型"""
        self.model = WaterQualityNet(
            input_size=input_size,
            hidden_sizes=[128, 256, 128],
            num_classes=len(self.label_encoder.classes_),
            dropout_rate=0.3
        ).to(self.device)

        print(f"🧠 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_model(self, train_loader, test_loader, epochs=100, learning_rate=0.001):
        """训练模型"""

        # 损失函数和优化器 (使用更温和的类别权重)
        class_counts = np.bincount(train_loader.dataset.y.numpy())
        max_count = np.max(class_counts)
        # 温和的类别权重：避免过度偏向少数类
        class_weights = max_count / (class_counts + 1e-6)
        class_weights = np.clip(class_weights, 0.5, 2.5)  # 限制权重范围

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        print(f"\n🚀 开始训练模型 ({epochs} 轮次)...")

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
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
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            # 记录历史
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), f'{self.output_dir}/best_model.pth')
            else:
                patience_counter += 1

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            # 早停
            if patience_counter >= patience:
                print(f"早停于第 {epoch+1} 轮")
                break

        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'{self.output_dir}/best_model.pth'))
        print("✅ 训练完成，已加载最佳模型")

    def evaluate_model(self, test_loader, X_test, y_test):
        """评估模型"""
        print("\n📊 评估模型性能...")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs, attention_weights = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(
            all_targets, all_predictions,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        print(f"📈 测试集准确率: {accuracy:.4f}")

        print("\n📊 各类别性能:")
        for i, label in enumerate(self.label_encoder.classes_):
            if label in report:
                precision = report[label]['precision']
                recall = report[label]['recall']
                f1 = report[label]['f1-score']
                support = report[label]['support']
                print(f"   {label:10s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, 支持数={support}")

        return all_targets, all_predictions

    def visualize_results(self, y_true, y_pred):
        """可视化结果"""
        print("\n📊 生成可视化结果...")

        # 1. 训练历史
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.title('损失变化')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='训练准确率')
        plt.plot(self.val_accuracies, label='验证准确率')
        plt.title('准确率变化')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True)

        # 2. 混淆矩阵
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/deep_learning_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 性能对比柱状图
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        metrics = ['precision', 'recall', 'f1-score']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            values = [report[label][metric] for label in self.label_encoder.classes_]
            axes[i].bar(self.label_encoder.classes_, values)
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_ylabel('分数')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_report(self, y_true, y_pred):
        """保存评估报告"""
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_
        )

        with open(f'{self.output_dir}/deep_learning_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("深度学习水质污染等级分类模型评估报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"模型训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用设备: {self.device}\n")
            f.write(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"输入特征数量: {len(self.feature_names)}\n")
            f.write(f"类别数量: {len(self.label_encoder.classes_)}\n\n")

            f.write("模型架构:\n")
            f.write("-" * 40 + "\n")
            f.write("- 多层感知机 (128-256-128)\n")
            f.write("- 自注意力机制\n")
            f.write("- BatchNorm + Dropout\n")
            f.write("- 加权交叉熵损失\n")
            f.write("- 早停 + 学习率调度\n\n")

            f.write("标签映射:\n")
            for i, label in self.label_mapping.items():
                f.write(f"  {i} -> {label}\n")
            f.write("\n")

            f.write("性能指标:\n")
            f.write("-" * 40 + "\n")
            f.write(report)
            f.write("\n")

            f.write("技术特点:\n")
            f.write("-" * 40 + "\n")
            f.write("1. ✅ GPU加速训练 (如果可用)\n")
            f.write("2. ✅ 自注意力机制增强特征学习\n")
            f.write("3. ✅ 类别不平衡处理 (加权采样)\n")
            f.write("4. ✅ 早停防止过拟合\n")
            f.write("5. ✅ 学习率自适应调整\n")
            f.write("6. ✅ 批量归一化和正则化\n")

        print(f"📄 评估报告已保存: {self.output_dir}/deep_learning_report.txt")

def main():
    """主函数"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch未安装，无法运行深度学习模型")
        return

    print("[启动] 深度学习水质分类模型训练...")

    # 创建分类器
    classifier = DeepLearningClassifier()

    # 加载和预处理数据（从项目根目录）
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'preprocessed_data_feature_eng.csv')
    print(f"[数据路径] {data_path}")
    X, y = classifier.load_and_preprocess_data(data_path)

    # 创建数据加载器
    train_loader, test_loader, X_test, y_test = classifier.create_data_loaders(X, y, batch_size=64)

    # 创建模型
    classifier.create_model(input_size=X.shape[1])

    # 训练模型
    classifier.train_model(train_loader, test_loader, epochs=100, learning_rate=0.001)

    # 评估模型
    y_true, y_pred = classifier.evaluate_model(test_loader, X_test, y_test)

    # 可视化结果
    classifier.visualize_results(y_true, y_pred)

    # 保存报告
    classifier.save_report(y_true, y_pred)

    print("\n🎉 深度学习模型训练完成!")
    print(f"📁 所有结果已保存到: {classifier.output_dir}")

if __name__ == "__main__":
    main()