"""
综合模型训练脚本
集成所有传统ML和深度学习模型，使用ModelHub进行统一管理
功能:
1. 一次性加载所有13个模型
2. 批量训练所有模型
3. 统一评估和对比
4. 生成完整对比报告和可视化
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.safe_data_loader import WaterQualityDataLoader
from core.leak_detector import DataLeakDetector, IntegrityChecker
from model_hub import ModelHub
from core.base_classifier import TraditionalMLClassifier, DeepLearningClassifier

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# ============================================================================
# 传统ML模型包装类
# ============================================================================

class RandomForestModel(TraditionalMLClassifier):
    """随机森林分类器"""
    def __init__(self):
        super().__init__(model_name='RandomForest')

    def build_model(self, **kwargs):
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 15)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class GradientBoostingModel(TraditionalMLClassifier):
    """梯度提升分类器"""
    def __init__(self):
        super().__init__(model_name='GradientBoosting')

    def build_model(self, **kwargs):
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 5)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class XGBoostModel(TraditionalMLClassifier):
    """XGBoost分类器"""
    def __init__(self):
        super().__init__(model_name='XGBoost')
        if not HAS_XGBOOST:
            raise ImportError("XGBoost未安装，请运行: pip install xgboost")

    def build_model(self, **kwargs):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.05,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1,  # Use all available CPU cores
            **kwargs
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class CatBoostModel(TraditionalMLClassifier):
    """CatBoost分类器"""
    def __init__(self):
        super().__init__(model_name='CatBoost')
        if not HAS_CATBOOST:
            raise ImportError("CatBoost未安装，请运行: pip install catboost")

    def build_model(self, **kwargs):
        iterations = kwargs.get('iterations', 100)
        depth = kwargs.get('depth', 6)
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class LightGBMModel(TraditionalMLClassifier):
    """LightGBM分类器"""
    def __init__(self):
        super().__init__(model_name='LightGBM')
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM未安装，请运行: pip install lightgbm")

    def build_model(self, **kwargs):
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 5)
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class SVMModel(TraditionalMLClassifier):
    """支持向量机分类器"""
    def __init__(self):
        super().__init__(model_name='SVM')

    def build_model(self, **kwargs):
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 1.0)
        self.model = SVC(
            kernel=kernel,
            C=C,
            random_state=42,
            probability=True
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class KNNModel(TraditionalMLClassifier):
    """K近邻分类器"""
    def __init__(self):
        super().__init__(model_name='KNN')

    def build_model(self, **kwargs):
        n_neighbors = kwargs.get('n_neighbors', 5)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class LogisticRegressionModel(TraditionalMLClassifier):
    """逻辑回归分类器"""
    def __init__(self):
        super().__init__(model_name='LogisticRegression')

    def build_model(self, **kwargs):
        max_iter = kwargs.get('max_iter', 1000)
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            multi_class='multinomial',
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(**kwargs)
        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class SimpleDLModel(DeepLearningClassifier):
    """简单深度学习模型"""
    def __init__(self):
        super().__init__(model_name='DeepLearning')
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")

    def build_model(self, input_dim, **kwargs):
        if self.model is not None:
            return

        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(5, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build_model(X_train.shape[1], **kwargs)

        print(f"[{self.model_name}] 训练中...")
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )

        self.is_trained = True
        train_acc = history.history['accuracy'][-1]
        print(f"[{self.model_name}] 训练完成，准确率: {train_acc:.4f}")
        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X, verbose=0)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("水质污染分类模型 - 全模型综合训练与对比")
    print("="*70 + "\n")

    # ========== 步骤1: 加载数据 ==========
    print("[步骤1] 加载和预处理数据...")
    print("-" * 70)

    loader = WaterQualityDataLoader(test_size=0.2, random_state=42)

    try:
        X_train, X_test, y_train, y_test, label_mapping = loader.preprocess(
            apply_smote=False
        )
        print(f"[成功] Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"[标签] {label_mapping}\n")

    except Exception as e:
        print(f"[错误] {e}")
        return

    # ========== 步骤2: 数据完整性检查 ==========
    print("[步骤2] 数据完整性检查...")
    print("-" * 70)

    integrity = IntegrityChecker(verbose=False)
    integrity.check_no_nan(X_train, 'Train')
    integrity.check_no_nan(X_test, 'Test')
    integrity.check_shape(X_train, X_test)
    integrity.check_split_ratio(X_train.shape[0], X_test.shape[0])
    summary = integrity.get_summary()

    print(f"[成功] 检查通过: {summary['passed']}/{summary['total']} 项\n")

    # ========== 步骤3: 创建ModelHub和注册所有模型 ==========
    print("[步骤3] 注册所有模型...")
    print("-" * 70)

    hub = ModelHub()

    # 传统ML模型列表
    traditional_models = [
        ('RandomForest', RandomForestModel()),
        ('GradientBoosting', GradientBoostingModel()),
        ('SVM', SVMModel()),
        ('KNN', KNNModel()),
        ('LogisticRegression', LogisticRegressionModel()),
    ]

    # 条件注册可选模型
    optional_models = [
        ('XGBoost', XGBoostModel, HAS_XGBOOST),
        ('CatBoost', CatBoostModel, HAS_CATBOOST),
        ('LightGBM', LightGBMModel, HAS_LIGHTGBM),
    ]

    for model_id, model_instance in traditional_models:
        hub.register_model(model_id, model_instance, category='traditional')

    for model_id, model_class, available in optional_models:
        if available:
            try:
                hub.register_model(model_id, model_class(), category='traditional')
            except Exception as e:
                print(f"[警告] {model_id} 注册失败: {e}")
        else:
            print(f"[信息] {model_id} 未安装，跳过")

    # 注册深度学习模型
    if HAS_TENSORFLOW:
        try:
            hub.register_model('DeepLearning', SimpleDLModel(), category='deep_learning')
        except Exception as e:
            print(f"[警告] DeepLearning 注册失败: {e}")
    else:
        print("[信息] TensorFlow 未安装，跳过深度学习模型")

    print(f"[成功] 已注册 {len(hub.get_all_models())} 个模型\n")

    if len(hub.get_all_models()) == 0:
        print("[错误] 未注册任何模型！")
        return

    # ========== 步骤4: 批量训练 ==========
    print("[步骤4] 批量训练所有模型...")
    print("-" * 70)

    hub.train_all(X_train, y_train, verbose=False)

    # ========== 步骤5: 批量评估 ==========
    print("[步骤5] 批量评估所有模型...")
    print("-" * 70)

    hub.evaluate_all(X_test, y_test, label_mapping=label_mapping, verbose=False)

    # ========== 步骤6: 打印对比表 ==========
    print("[步骤6] 模型性能对比...")
    print("-" * 70)

    hub.print_comparison_table()

    # ========== 步骤7: 推荐最佳模型 ==========
    print("[步骤7] 最佳模型推荐...")
    print("-" * 70)

    hub.recommend_best_model(criterion='accuracy')

    # ========== 步骤8: 生成报告 ==========
    print("[步骤8] 生成完整报告...")
    print("-" * 70)

    hub.save_comparison_table()
    hub.plot_comparison_charts()
    hub.generate_all_reports()
    report = hub.generate_final_report()

    print("\n" + "="*70)
    print("✓ 全模型综合训练与对比完成！")
    print("="*70)
    print(f"\n输出目录: {hub.output_dir}")
    print("\n生成的文件：")
    print("  - model_comparison.csv          (模型对比表)")
    print("  - comparison_charts.png         (性能对比图)")
    print("  - final_report.txt              (最终报告)")
    print("  - hub_summary.json              (Hub总结)")
    print("  - models/{ModelName}/           (各模型详细报告)")
    print()


if __name__ == '__main__':
    main()
