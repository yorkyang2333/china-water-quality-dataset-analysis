#!/usr/bin/env python3
"""
混合集成学习水质分类系统
结合深度学习、传统机器学习和集成学习方法

策略:
1. 深度学习模型 (PyTorch MLP)
2. 传统机器学习模型 (XGBoost, RandomForest, LightGBM, CatBoost)
3. 集成学习 (Stacking, Voting, Blending)
4. 专门针对少数类的优化策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import joblib
from itertools import combinations

# PyTorch相关
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# 传统机器学习模型
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

# 数据平衡
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FocalLoss(nn.Module):
    """焦点损失"""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

class WaterQualityDataset(Dataset):
    """PyTorch数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleMLP(nn.Module):
    """简化的MLP用于集成"""
    def __init__(self, input_size, hidden_sizes=[256, 128], num_classes=5, dropout_rate=0.3):
        super(SimpleMLP, self).__init__()

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

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class HybridEnsembleClassifier:
    """混合集成分类器"""

    def __init__(self, output_dir='Outputs/Hybrid_Ensemble_Classification'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

        # 存储各个模型
        self.models = {}
        self.ensemble_model = None

        print(f"🔧 使用设备: {self.device}")
        print(f"📦 可用库状态: PyTorch={TORCH_AVAILABLE}, XGBoost={XGB_AVAILABLE}, "
              f"LightGBM={LGB_AVAILABLE}, CatBoost={CAT_AVAILABLE}")

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
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

        # 周期性编码
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        # 地理特征编码
        df['Province_encoded'] = df['Province'].astype('category').cat.codes
        df['City_encoded'] = df['City'].astype('category').cat.codes
        df['Station_encoded'] = df['Monitoring_Station'].astype('category').cat.codes

        # 处理缺失值
        for col in ['NP_Ratio', 'COD_BOD_Ratio']:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)

        # 特征选择
        feature_cols = [
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm',
            'Turbidity_NTU', 'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
            'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
            'Coliform_Count_CFU_100mL',
            'Total_Heavy_Metals_ug_L', 'COD_BOD_Ratio', 'NP_Ratio',
            'Latitude', 'Longitude', 'Province_encoded', 'City_encoded', 'Station_encoded',
            'Year', 'Month', 'Season', 'DayOfWeek', 'IsWeekend',
            'Month_sin', 'Month_cos'
        ]

        # 移除不需要的列
        drop_cols = ['Province', 'City', 'Monitoring_Station', 'Date', 'Remarks', 'Water_Quality_Index']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # 确保所有特征列都存在
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features

        # 准备特征和目标
        X = df[available_features].copy()
        y = df['Pollution_Level'].copy()

        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}

        print(f"📊 标签映射: {self.label_mapping}")
        print(f"✅ 使用 {len(available_features)} 个特征")

        return X, y_encoded

    def balance_data_multiple_methods(self, X, y):
        """使用多种方法平衡数据"""
        print("\n🔄 创建多个平衡数据集...")

        balanced_datasets = {}

        # 1. 原始数据
        balanced_datasets['original'] = (X, y)

        # 2. SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        balanced_datasets['smote'] = (X_smote, y_smote)

        # 3. ADASYN
        adasyn = ADASYN(random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
        balanced_datasets['adasyn'] = (X_adasyn, y_adasyn)

        # 4. BorderlineSMOTE
        borderline_smote = BorderlineSMOTE(random_state=42)
        X_borderline, y_borderline = borderline_smote.fit_resample(X, y)
        balanced_datasets['borderline'] = (X_borderline, y_borderline)

        print(f"✅ 创建了 {len(balanced_datasets)} 个平衡数据集")

        return balanced_datasets

    def train_individual_models(self, X_train, y_train, X_test, y_test):
        """训练各个独立模型"""
        print("\n🚀 开始训练各个独立模型...")

        # 数据分割和标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        # 1. 深度学习模型
        if TORCH_AVAILABLE:
            print("\n🧠 训练深度学习模型...")
            dl_results = self.train_deep_learning_model(X_train_scaled, y_train, X_test_scaled, y_test)
            results['deep_learning'] = dl_results

        # 2. XGBoost
        if XGB_AVAILABLE:
            print("\n🌲 训练XGBoost模型...")
            xgb_results = self.train_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test)
            results['xgboost'] = xgb_results

        # 3. LightGBM
        if LGB_AVAILABLE:
            print("\n💡 训练LightGBM模型...")
            lgb_results = self.train_lightgbm_model(X_train_scaled, y_train, X_test_scaled, y_test)
            results['lightgbm'] = lgb_results

        # 4. CatBoost
        if CAT_AVAILABLE:
            print("\n🐱 训练CatBoost模型...")
            cat_results = self.train_catboost_model(X_train_scaled, y_train, X_test_scaled, y_test)
            results['catboost'] = cat_results

        # 5. 平衡随机森林
        print("\n⚖️ 训练平衡随机森林模型...")
        brf_results = self.train_balanced_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
        results['balanced_rf'] = brf_results

        # 6. 容易集成分类器
        print("\n🎯 训练EasyEnsemble模型...")
        ee_results = self.train_easy_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)
        results['easy_ensemble'] = ee_results

        self.models = results
        return results

    def train_deep_learning_model(self, X_train, y_train, X_test, y_test):
        """训练深度学习模型"""
        # 数据平衡
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # 创建数据集和加载器
        train_dataset = WaterQualityDataset(X_train_balanced, y_train_balanced)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 创建模型
        model = SimpleMLP(
            input_size=X_train.shape[1],
            hidden_sizes=[256, 128],
            num_classes=len(self.label_encoder.classes_),
            dropout_rate=0.3
        ).to(self.device)

        # 训练设置
        class_counts = np.bincount(y_train_balanced)
        class_weights = torch.tensor(len(class_counts) / class_counts, dtype=torch.float32).to(self.device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练
        model.train()
        for epoch in range(50):  # 快速训练
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 预测
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()

        # 保存模型
        torch.save(model.state_dict(), f'{self.output_dir}/dl_model.pth')

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': F.softmax(outputs, dim=1).cpu().numpy()
        }

    def train_xgboost_model(self, X_train, y_train, X_test, y_test):
        """训练XGBoost模型"""
        # 计算类别权重
        class_counts = np.bincount(y_train)
        scale_pos_weight = {i: class_counts[0]/class_counts[i] for i in range(len(class_counts))}

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            tree_method='hist'
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 保存模型
        joblib.dump(model, f'{self.output_dir}/xgb_model.joblib')

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def train_lightgbm_model(self, X_train, y_train, X_test, y_test):
        """训练LightGBM模型"""
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 保存模型
        joblib.dump(model, f'{self.output_dir}/lgb_model.joblib')

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def train_catboost_model(self, X_train, y_train, X_test, y_test):
        """训练CatBoost模型"""
        model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
            class_weights=[len(y_train)/np.sum(y_train==i) for i in range(len(np.unique(y_train)))]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 保存模型
        joblib.dump(model, f'{self.output_dir}/cat_model.joblib')

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def train_balanced_random_forest(self, X_train, y_train, X_test, y_test):
        """训练平衡随机森林"""
        model = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            sampling_strategy='auto',
            replacement=True,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 保存模型
        joblib.dump(model, f'{self.output_dir}/brf_model.joblib')

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def train_easy_ensemble(self, X_train, y_train, X_test, y_test):
        """训练EasyEnsemble分类器"""
        model = EasyEnsembleClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 保存模型
        joblib.dump(model, f'{self.output_dir}/ee_model.joblib')

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def create_voting_ensemble(self, y_true):
        """创建投票集成模型"""
        print("\n🗳️ 创建投票集成模型...")

        if not self.models:
            raise ValueError("需要先训练独立模型")

        # 收集所有预测
        all_predictions = []
        all_probabilities = []
        model_names = []

        for name, results in self.models.items():
            if 'predictions' in results:
                all_predictions.append(results['predictions'])
                all_probabilities.append(results['probabilities'])
                model_names.append(name)

        # 软投票（基于概率）
        avg_probabilities = np.mean(all_probabilities, axis=0)
        voting_predictions = np.argmax(avg_probabilities, axis=1)

        return {
            'voting_predictions': voting_predictions,
            'voting_probabilities': avg_probabilities,
            'individual_predictions': all_predictions,
            'individual_probabilities': all_probabilities,
            'model_names': model_names
        }

    def create_stacking_ensemble(self, X_train, y_train, X_test, y_test):
        """创建堆叠集成模型"""
        print("\n📚 创建堆叠集成模型...")

        # 准备基模型
        base_models = []

        if XGB_AVAILABLE:
            xgb_base = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, eval_metric='mlogloss', verbose=False
            )
            base_models.append(('xgb', xgb_base))

        if LGB_AVAILABLE:
            lgb_base = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, class_weight='balanced', verbose=-1
            )
            base_models.append(('lgb', lgb_base))

        # 平衡随机森林
        rf_base = BalancedRandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        base_models.append(('balanced_rf', rf_base))

        # 元模型
        meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')

        # 创建堆叠分类器
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1
        )

        # 训练
        stacking_model.fit(X_train, y_train)
        stacking_pred = stacking_model.predict(X_test)
        stacking_proba = stacking_model.predict_proba(X_test)

        # 保存模型
        joblib.dump(stacking_model, f'{self.output_dir}/stacking_model.joblib')

        return {
            'model': stacking_model,
            'predictions': stacking_pred,
            'probabilities': stacking_proba
        }

    def optimize_thresholds(self, y_true, y_proba):
        """优化分类阈值"""
        print("\n🎯 优化分类阈值...")

        n_classes = len(self.label_encoder.classes_)
        optimized_predictions = np.zeros(len(y_true))

        # 对每个类别优化阈值
        for i in range(n_classes):
            # 尝试不同的阈值
            best_threshold = 1.0 / n_classes  # 默认阈值
            best_f1 = 0

            for threshold in np.linspace(0.05, 0.5, 20):
                test_pred = (y_proba[:, i] >= threshold).astype(int)
                if np.sum(test_pred) > 0:  # 确保至少有一个预测
                    from sklearn.metrics import f1_score
                    binary_true = (y_true == i).astype(int)
                    f1 = f1_score(binary_true, test_pred, zero_division=0)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

            # 应用最优阈值
            optimized_predictions[y_proba[:, i] >= best_threshold] = i

        return optimized_predictions

    def evaluate_all_models(self, y_true):
        """评估所有模型"""
        print("\n📊 评估所有模型性能...")

        results_summary = {}

        # 评估独立模型
        print("\n独立模型性能:")
        print("-" * 60)
        for name, results in self.models.items():
            if 'predictions' in results:
                y_pred = results['predictions']
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(
                    y_true, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )

                results_summary[name] = {
                    'accuracy': accuracy,
                    'macro_f1': report['macro avg']['f1-score'],
                    'weighted_f1': report['weighted avg']['f1-score']
                }

                print(f"{name:15s}: 准确率={accuracy:.4f}, 宏F1={report['macro avg']['f1-score']:.4f}")

                # 特别关注Poor和Very Poor的性能
                if 'Poor' in report and 'Very Poor' in report:
                    poor_f1 = report['Poor']['f1-score']
                    very_poor_f1 = report['Very Poor']['f1-score']
                    print(f"{'':15s}  Poor F1={poor_f1:.3f}, Very Poor F1={very_poor_f1:.3f}")

        return results_summary

    def visualize_results(self, y_true, ensemble_results, results_summary):
        """可视化结果"""
        print("\n📊 生成可视化结果...")

        # 1. 模型性能对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 准确率对比
        ax1 = axes[0, 0]
        models = list(results_summary.keys())
        accuracies = [results_summary[m]['accuracy'] for m in models]
        bars = ax1.bar(models, accuracies)
        ax1.set_title('各模型准确率对比')
        ax1.set_ylabel('准确率')
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')

        # 宏F1对比
        ax2 = axes[0, 1]
        macro_f1s = [results_summary[m]['macro_f1'] for m in models]
        bars = ax2.bar(models, macro_f1s)
        ax2.set_title('各模型宏平均F1对比')
        ax2.set_ylabel('宏平均F1')
        ax2.tick_params(axis='x', rotation=45)
        for bar, f1 in zip(bars, macro_f1s):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{f1:.3f}', ha='center', va='bottom')

        # 混淆矩阵 - 投票集成
        ax3 = axes[1, 0]
        cm = confusion_matrix(y_true, ensemble_results['voting_predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=ax3)
        ax3.set_title('投票集成混淆矩阵')
        ax3.set_xlabel('预测标签')
        ax3.set_ylabel('真实标签')

        # 各类别详细性能
        ax4 = axes[1, 1]
        voting_report = classification_report(
            y_true, ensemble_results['voting_predictions'],
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        categories = []
        precisions = []
        recalls = []
        f1_scores = []

        for label in self.label_encoder.classes_:
            if label in voting_report:
                categories.append(label)
                precisions.append(voting_report[label]['precision'])
                recalls.append(voting_report[label]['recall'])
                f1_scores.append(voting_report[label]['f1-score'])

        x = np.arange(len(categories))
        width = 0.25

        ax4.bar(x - width, precisions, width, label='精确率', alpha=0.8)
        ax4.bar(x, recalls, width, label='召回率', alpha=0.8)
        ax4.bar(x + width, f1_scores, width, label='F1分数', alpha=0.8)

        ax4.set_xlabel('类别')
        ax4.set_ylabel('分数')
        ax4.set_title('投票集成各类别性能')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hybrid_ensemble_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_comprehensive_report(self, y_true, ensemble_results, results_summary):
        """保存综合报告"""
        print("\n📄 保存综合评估报告...")

        with open(f'{self.output_dir}/hybrid_ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("混合集成学习水质污染等级分类系统综合评估报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"系统设备: {self.device}\n")
            f.write(f"特征数量: {len(self.feature_names)}\n")
            f.write(f"类别数量: {len(self.label_encoder.classes_)}\n\n")

            f.write("标签映射:\n")
            for i, label in self.label_mapping.items():
                f.write(f"  {i} -> {label}\n")
            f.write("\n")

            f.write("集成策略:\n")
            f.write("-" * 40 + "\n")
            f.write("1. 🧠 深度学习模型 (PyTorch MLP)\n")
            f.write("2. 🌲 XGBoost 梯度提升\n")
            f.write("3. 💡 LightGBM 轻量级梯度提升\n")
            f.write("4. 🐱 CatBoost 类别特征优化\n")
            f.write("5. ⚖️ 平衡随机森林\n")
            f.write("6. 🎯 EasyEnsemble 集成\n")
            f.write("7. 🗳️ 软投票集成\n")
            f.write("8. 📚 堆叠集成\n")
            f.write("9. 🎯 阈值优化\n\n")

            f.write("独立模型性能排名:\n")
            f.write("-" * 40 + "\n")
            sorted_results = sorted(results_summary.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
            for i, (name, metrics) in enumerate(sorted_results, 1):
                f.write(f"{i:2d}. {name:20s}: 准确率={metrics['accuracy']:.4f}, 宏F1={metrics['macro_f1']:.4f}\n")
            f.write("\n")

            # 投票集成详细性能
            voting_report = classification_report(
                y_true, ensemble_results['voting_predictions'],
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
            f.write("投票集成详细性能:\n")
            f.write("-" * 40 + "\n")
            f.write(voting_report)
            f.write("\n")

            # 特别关注Poor和Very Poor
            voting_dict = classification_report(
                y_true, ensemble_results['voting_predictions'],
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )

            f.write("重点关注类别 (Poor & Very Poor) 性能:\n")
            f.write("-" * 40 + "\n")
            if 'Poor' in voting_dict:
                f.write(f"Poor      : P={voting_dict['Poor']['precision']:.3f}, R={voting_dict['Poor']['recall']:.3f}, F1={voting_dict['Poor']['f1-score']:.3f}\n")
            if 'Very Poor' in voting_dict:
                f.write(f"Very Poor : P={voting_dict['Very Poor']['precision']:.3f}, R={voting_dict['Very Poor']['recall']:.3f}, F1={voting_dict['Very Poor']['f1-score']:.3f}\n")
            f.write("\n")

            f.write("技术创新点:\n")
            f.write("-" * 40 + "\n")
            f.write("✅ 多模型融合提升鲁棒性\n")
            f.write("✅ 专门针对类别不平衡的多种策略\n")
            f.write("✅ 软投票和堆叠双重集成\n")
            f.write("✅ 自适应阈值优化\n")
            f.write("✅ 深度学习与传统机器学习结合\n")

        print(f"📄 综合报告已保存: {self.output_dir}/hybrid_ensemble_report.txt")

def main():
    """主函数"""
    print("🚀 启动混合集成学习水质分类系统...")

    # 创建混合集成分类器
    classifier = HybridEnsembleClassifier()

    # 加载数据
    data_path = 'Outputs/Data/preprocessed_data_feature_eng.csv'
    X, y = classifier.load_and_preprocess_data(data_path)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 数据分割: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")

    # 训练独立模型
    individual_results = classifier.train_individual_models(X_train, y_train, X_test, y_test)

    # 创建投票集成
    voting_results = classifier.create_voting_ensemble(y_test)

    # 创建堆叠集成
    try:
        stacking_results = classifier.create_stacking_ensemble(X_train, y_train, X_test, y_test)
        classifier.models['stacking'] = stacking_results
    except Exception as e:
        print(f"堆叠集成创建失败: {e}")

    # 优化阈值
    optimized_predictions = classifier.optimize_thresholds(y_test, voting_results['voting_probabilities'])

    # 评估所有模型
    results_summary = classifier.evaluate_all_models(y_test)

    # 添加投票集成结果
    voting_accuracy = accuracy_score(y_test, voting_results['voting_predictions'])
    voting_report = classification_report(
        y_test, voting_results['voting_predictions'],
        target_names=classifier.label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    results_summary['voting_ensemble'] = {
        'accuracy': voting_accuracy,
        'macro_f1': voting_report['macro avg']['f1-score'],
        'weighted_f1': voting_report['weighted avg']['f1-score']
    }

    # 添加优化阈值结果
    optimized_accuracy = accuracy_score(y_test, optimized_predictions)
    optimized_report = classification_report(
        y_test, optimized_predictions,
        target_names=classifier.label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    results_summary['optimized_threshold'] = {
        'accuracy': optimized_accuracy,
        'macro_f1': optimized_report['macro avg']['f1-score'],
        'weighted_f1': optimized_report['weighted avg']['f1-score']
    }

    # 可视化结果
    classifier.visualize_results(y_test, voting_results, results_summary)

    # 保存综合报告
    voting_results['optimized_predictions'] = optimized_predictions
    classifier.save_comprehensive_report(y_test, voting_results, results_summary)

    print("\n🎉 混合集成学习系统训练完成!")
    print(f"📁 所有结果已保存到: {classifier.output_dir}")

    # 打印最佳模型
    best_model = max(results_summary.items(), key=lambda x: x[1]['macro_f1'])
    print(f"\n🏆 最佳模型: {best_model[0]} (宏F1: {best_model[1]['macro_f1']:.4f})")

if __name__ == "__main__":
    main()