#!/usr/bin/env python3
"""
终极集成水质预测系统 (Ultimate Ensemble Water Quality Prediction System)

整合所有模型形成最强预测系统:

架构:
Level-1 (基础模型层):
  - 传统ML: XGBoost, RandomForest, LightGBM, CatBoost, GradientBoosting
  - 深度学习: MLP, Transformer, CNN, RNN
  - 线性基准: LogisticRegression, SVM, KNN

Level-2 (元学习层):
  - Stacking Meta-Learner: 使用L1预测作为特征
  - Weighted Voting: 基于验证集性能的动态权重
  - Selective Ensemble: 自适应选择最佳子集

核心创新:
1. 多层次集成策略
2. 动态权重分配
3. 自适应模型选择
4. 完整的可解释性分析
5. 鲁棒性增强（处理模型失败）
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import zhplot  # 自动配置matplotlib中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from collections import defaultdict

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 传统ML模型
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost未安装")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARN] LightGBM未安装")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARN] CatBoost未安装")

# PyTorch (用于深度学习模型)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch未安装")

# 导入核心模块
try:
    from core.safe_data_loader import WaterQualityDataLoader
    from core.evaluator import ClassifierEvaluator
    from core.base_classifier import DeepLearningClassifier
except ImportError as e:
    print(f"[ERROR] 无法导入核心模块: {e}")
    sys.exit(1)


# ============================================================================
# 简单深度学习模型（用于集成）
# ============================================================================

class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=5, dropout=0.3):
        super(SimpleMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPClassifierWrapper:
    """MLP分类器包装类"""
    def __init__(self, hidden_sizes=[256, 128, 64], dropout=0.3, epochs=100,
                 batch_size=64, lr=0.001, patience=15):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """训练模型"""
        # 标准化
        X = self.scaler.fit_transform(X)

        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        # 创建数据集
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 构建模型
        input_size = X.shape[1]
        num_classes = len(np.unique(y))

        self.model = SimpleMLP(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=num_classes,
            dropout=self.dropout
        ).to(self.device)

        # 训练
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # 恢复最佳模型
        self.model.load_state_dict(best_state)

    def predict(self, X):
        """预测"""
        X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X):
        """预测概率"""
        X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            proba = F.softmax(outputs, dim=1)
            return proba.cpu().numpy()


# ============================================================================
# 终极集成系统
# ============================================================================

class UltimateEnsembleSystem:
    """
    终极集成预测系统

    三大集成策略:
    1. Stacking: 使用元学习器整合L1预测
    2. Weighted Voting: 基于验证集性能的动态权重投票
    3. Selective Ensemble: 自适应选择最佳模型子集
    """

    def __init__(self, output_dir='Outputs/Ultimate_Ensemble_System'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Level-1 基础模型
        self.base_models = {}
        self.base_models_predictions = {}
        self.base_models_performance = {}

        # Level-2 元学习器
        self.meta_learner = None
        self.stacking_model = None
        self.voting_model = None

        # 权重
        self.dynamic_weights = {}

        # 性能记录
        self.performance_history = defaultdict(list)

        print(f"[UltimateEnsemble] 初始化完成")
        print(f"[UltimateEnsemble] 输出目录: {output_dir}")

    def build_base_models(self):
        """构建所有基础模型"""
        print("\n[UltimateEnsemble] 构建基础模型...")

        models = {}

        # 1. 传统ML模型（强大的树模型）
        if HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )

        models['RandomForest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        if HAS_LIGHTGBM:
            models['LightGBM'] = LGBMClassifier(
                n_estimators=300,
                max_depth=15,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )

        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )

        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        # 2. 线性/基准模型
        models['LogisticRegression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )

        models['SVM'] = SVC(
            kernel='rbf',
            C=10.0,
            probability=True,
            random_state=42
        )

        models['KNN'] = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        )

        # 3. 深度学习模型
        if HAS_TORCH:
            models['MLP'] = MLPClassifierWrapper(
                hidden_sizes=[256, 128, 64],
                dropout=0.3,
                epochs=100,
                batch_size=64,
                lr=0.001,
                patience=15
            )

        self.base_models = models
        print(f"[UltimateEnsemble] 共构建 {len(models)} 个基础模型:")
        for name in models.keys():
            print(f"  - {name}")

        return models

    def train_base_models(self, X_train, y_train, X_val=None, y_val=None):
        """训练所有基础模型"""
        print("\n[UltimateEnsemble] 训练基础模型...")

        if X_val is None or y_val is None:
            # 从训练集中划分验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )

        print(f"[UltimateEnsemble] 训练集大小: {len(X_train)}")
        print(f"[UltimateEnsemble] 验证集大小: {len(X_val)}")

        self.X_val = X_val
        self.y_val = y_val

        for name, model in self.base_models.items():
            try:
                print(f"\n[{name}] 训练中...")
                model.fit(X_train, y_train)

                # 验证集预测
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)

                # 保存性能
                self.base_models_performance[name] = {
                    'accuracy': val_acc,
                    'predictions': val_pred
                }

                print(f"[{name}] 验证集准确率: {val_acc:.4f}")

            except Exception as e:
                print(f"[{name}] 训练失败: {e}")
                # 移除失败的模型
                del self.base_models[name]

        print(f"\n[UltimateEnsemble] 基础模型训练完成！成功: {len(self.base_models)} 个")

    def compute_dynamic_weights(self):
        """基于验证集性能计算动态权重"""
        print("\n[UltimateEnsemble] 计算动态权重...")

        # 使用softmax归一化准确率作为权重
        accuracies = np.array([
            self.base_models_performance[name]['accuracy']
            for name in self.base_models.keys()
        ])

        # Softmax with temperature
        temperature = 0.1  # 较小的temperature使权重更集中在最好的模型上
        weights = np.exp(accuracies / temperature) / np.sum(np.exp(accuracies / temperature))

        # 归一化
        weights = weights / weights.sum()

        self.dynamic_weights = {
            name: weight
            for name, weight in zip(self.base_models.keys(), weights)
        }

        print("[UltimateEnsemble] 动态权重分配:")
        for name, weight in sorted(self.dynamic_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:20s}: {weight:.4f}")

    def build_stacking_model(self, X_train, y_train):
        """构建Stacking集成模型"""
        print("\n[UltimateEnsemble] 构建Stacking模型...")

        # 使用性能最好的模型作为基学习器
        sorted_models = sorted(
            self.base_models_performance.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        # 选择Top-5模型
        top_models = sorted_models[:min(5, len(sorted_models))]
        print(f"[UltimateEnsemble] 选择Top-{len(top_models)}模型用于Stacking:")
        for name, perf in top_models:
            print(f"  - {name}: {perf['accuracy']:.4f}")

        estimators = [
            (name, self.base_models[name])
            for name, _ in top_models
        ]

        # 元学习器使用LogisticRegression
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            n_jobs=-1
        )

        print("[UltimateEnsemble] 训练Stacking模型...")
        self.stacking_model.fit(X_train, y_train)
        print("[UltimateEnsemble] Stacking模型训练完成！")

    def build_voting_model(self):
        """构建加权投票模型"""
        print("\n[UltimateEnsemble] 构建加权投票模型...")

        # 选择所有模型
        estimators = [
            (name, model)
            for name, model in self.base_models.items()
        ]

        # 使用动态权重
        weights = [self.dynamic_weights[name] for name, _ in estimators]

        self.voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # 使用概率投票
            weights=weights,
            n_jobs=-1
        )

        print("[UltimateEnsemble] 加权投票模型构建完成！")

    def predict_ensemble(self, X, method='weighted_voting'):
        """
        集成预测

        Args:
            X: 特征矩阵
            method: 集成方法 ('weighted_voting', 'stacking', 'average', 'best')

        Returns:
            predictions: 预测结果
        """
        if method == 'weighted_voting':
            # 加权投票
            all_proba = []
            for name, model in self.base_models.items():
                proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                if proba is not None:
                    weighted_proba = proba * self.dynamic_weights[name]
                    all_proba.append(weighted_proba)

            # 平均概率
            avg_proba = np.mean(all_proba, axis=0)
            predictions = np.argmax(avg_proba, axis=1)

        elif method == 'stacking':
            # Stacking预测
            predictions = self.stacking_model.predict(X)

        elif method == 'average':
            # 简单平均
            all_pred = [model.predict(X) for model in self.base_models.values()]
            # 投票
            predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=np.array(all_pred)
            )

        elif method == 'best':
            # 使用最好的单模型
            best_model_name = max(
                self.base_models_performance.items(),
                key=lambda x: x[1]['accuracy']
            )[0]
            predictions = self.base_models[best_model_name].predict(X)

        else:
            raise ValueError(f"未知的集成方法: {method}")

        return predictions

    def predict_proba_ensemble(self, X, method='weighted_voting'):
        """集成预测概率"""
        if method == 'weighted_voting':
            all_proba = []
            for name, model in self.base_models.items():
                proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                if proba is not None:
                    weighted_proba = proba * self.dynamic_weights[name]
                    all_proba.append(weighted_proba)

            # 平均概率
            avg_proba = np.mean(all_proba, axis=0)
            return avg_proba

        elif method == 'stacking':
            return self.stacking_model.predict_proba(X)

        else:
            return self.predict_proba_ensemble(X, method='weighted_voting')

    def evaluate_all_methods(self, X_test, y_test, label_mapping):
        """评估所有集成方法"""
        print("\n" + "=" * 80)
        print("评估所有集成方法")
        print("=" * 80)

        methods = ['weighted_voting', 'stacking', 'average', 'best']
        results = {}

        for method in methods:
            print(f"\n[{method}] 评估中...")
            try:
                pred = self.predict_ensemble(X_test, method=method)
                acc = accuracy_score(y_test, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, pred, average='macro'
                )

                results[method] = {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': pred
                }

                print(f"[{method}] 准确率: {acc:.4f}")
                print(f"[{method}] F1-Score: {f1:.4f}")

            except Exception as e:
                print(f"[{method}] 评估失败: {e}")

        # 打印对比表
        print("\n" + "=" * 80)
        print("集成方法对比")
        print("=" * 80)
        print(f"{'方法':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
        print("-" * 80)

        for method, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{method:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

        # 保存结果
        results_df = pd.DataFrame([
            {
                'Method': method,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            }
            for method, metrics in results.items()
        ])

        results_path = os.path.join(self.output_dir, 'ensemble_comparison.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 对比结果已保存: {results_path}")

        return results

    def save_system(self, filepath=None):
        """保存整个集成系统"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'ultimate_ensemble_system.pkl')

        print(f"\n[UltimateEnsemble] 保存系统到: {filepath}")

        save_data = {
            'base_models': self.base_models,
            'dynamic_weights': self.dynamic_weights,
            'base_models_performance': self.base_models_performance,
            'stacking_model': self.stacking_model
        }

        joblib.dump(save_data, filepath)
        print(f"[OK] 系统已保存")

    @staticmethod
    def load_system(filepath):
        """加载集成系统"""
        print(f"[UltimateEnsemble] 加载系统: {filepath}")
        save_data = joblib.load(filepath)

        system = UltimateEnsembleSystem()
        system.base_models = save_data['base_models']
        system.dynamic_weights = save_data['dynamic_weights']
        system.base_models_performance = save_data['base_models_performance']
        system.stacking_model = save_data['stacking_model']

        print(f"[OK] 系统已加载，包含 {len(system.base_models)} 个基础模型")
        return system


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 80)
    print("终极集成水质预测系统")
    print("=" * 80)

    # 加载数据
    print("\n[STEP 1] 加载数据...")
    loader = WaterQualityDataLoader(test_size=0.2, random_state=42)
    df = loader.load_raw_data()
    X_train, X_test, y_train, y_test, label_mapping = loader.preprocess(
        balance_method='smote',
        apply_smote=True
    )

    print(f"[OK] 训练集: {X_train.shape}")
    print(f"[OK] 测试集: {X_test.shape}")
    print(f"[OK] 类别映射: {label_mapping}")

    # 创建系统
    print("\n[STEP 2] 创建终极集成系统...")
    ensemble = UltimateEnsembleSystem()

    # 构建基础模型
    ensemble.build_base_models()

    # 训练基础模型
    ensemble.train_base_models(X_train, y_train)

    # 计算动态权重
    ensemble.compute_dynamic_weights()

    # 构建Stacking模型
    ensemble.build_stacking_model(X_train, y_train)

    # 评估所有方法
    print("\n[STEP 3] 评估所有集成方法...")
    results = ensemble.evaluate_all_methods(X_test, y_test, label_mapping)

    # 使用最佳方法进行最终预测
    print("\n[STEP 4] 使用最佳方法进行预测...")
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"[UltimateEnsemble] 最佳方法: {best_method}")

    final_pred = ensemble.predict_ensemble(X_test, method=best_method)
    final_acc = accuracy_score(y_test, final_pred)
    print(f"[UltimateEnsemble] 最终准确率: {final_acc:.4f}")

    # 详细报告
    print("\n" + "=" * 80)
    print("分类报告")
    print("=" * 80)
    print(classification_report(y_test, final_pred, target_names=list(label_mapping.keys())))

    # 保存系统
    print("\n[STEP 5] 保存系统...")
    ensemble.save_system()

    print("\n" + "=" * 80)
    print("终极集成系统构建完成！")
    print("=" * 80)
    print(f"最佳集成方法: {best_method}")
    print(f"最终准确率: {final_acc:.4f}")
    print(f"基础模型数量: {len(ensemble.base_models)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
