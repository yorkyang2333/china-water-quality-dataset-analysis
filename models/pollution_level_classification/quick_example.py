"""
快速示例脚本
展示如何使用新框架进行模型训练、评估和对比

这个脚本演示：
1. 加载和预处理数据
2. 创建简单的模型
3. 使用ModelHub进行管理
4. 生成对比报告
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.safe_data_loader import WaterQualityDataLoader
from core.leak_detector import DataLeakDetector, IntegrityChecker
from model_hub import ModelHub
from core.base_classifier import TraditionalMLClassifier

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class SimpleRandomForestModel(TraditionalMLClassifier):
    """简单的随机森林分类器"""

    def __init__(self):
        super().__init__(model_name='RandomForest')

    def build_model(self, **kwargs):
        """构建模型"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练模型"""
        if self.model is None:
            self.build_model(**kwargs)

        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # 计算训练准确率
        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，训练准确率: {train_acc:.4f}")

        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        """预测"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class SimpleGradientBoostingModel(TraditionalMLClassifier):
    """简单的梯度提升分类器"""

    def __init__(self):
        super().__init__(model_name='GradientBoosting')

    def build_model(self, **kwargs):
        """构建模型"""
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练模型"""
        if self.model is None:
            self.build_model(**kwargs)

        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，训练准确率: {train_acc:.4f}")

        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        """预测"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class SimpleSVMModel(TraditionalMLClassifier):
    """简单的SVM分类器"""

    def __init__(self):
        super().__init__(model_name='SVM')

    def build_model(self, **kwargs):
        """构建模型"""
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练模型"""
        if self.model is None:
            self.build_model(**kwargs)

        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，训练准确率: {train_acc:.4f}")

        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        """预测"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class SimpleLogisticRegressionModel(TraditionalMLClassifier):
    """简单的逻辑回归分类器"""

    def __init__(self):
        super().__init__(model_name='LogisticRegression')

    def build_model(self, **kwargs):
        """构建模型"""
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练模型"""
        if self.model is None:
            self.build_model(**kwargs)

        print(f"[{self.model_name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        print(f"[{self.model_name}] 训练完成，训练准确率: {train_acc:.4f}")

        self.training_history = {'train_accuracy': train_acc}

    def predict(self, X):
        """预测"""
        if self.model is None or not self.is_trained:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("水质污染分类模型 - 快速示例")
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

    # ========== 步骤3: 创建ModelHub和注册模型 ==========
    print("[步骤3] 创建并注册模型...")
    print("-" * 70)

    hub = ModelHub()

    # 注册4个简单模型作为演示
    models = [
        ('RandomForest', SimpleRandomForestModel()),
        ('GradientBoosting', SimpleGradientBoostingModel()),
        ('SVM', SimpleSVMModel()),
        ('LogisticRegression', SimpleLogisticRegressionModel()),
    ]

    for model_id, model_instance in models:
        hub.register_model(model_id, model_instance, category='traditional')

    print(f"[成功] 已注册 {len(hub.get_all_models())} 个模型\n")

    # ========== 步骤4: 批量训练 ==========
    print("[步骤4] 批量训练模型...")
    print("-" * 70)

    hub.train_all(X_train, y_train, verbose=False)

    print()

    # ========== 步骤5: 批量评估 ==========
    print("[步骤5] 批量评估模型...")
    print("-" * 70)

    hub.evaluate_all(X_test, y_test, label_mapping=label_mapping, verbose=False)

    print()

    # ========== 步骤6: 打印对比表 ==========
    print("[步骤6] 模型性能对比...")
    print("-" * 70)

    hub.print_comparison_table()

    # ========== 步骤7: 推荐最佳模型 ==========
    print("[步骤7] 最佳模型推荐...")
    print("-" * 70)

    hub.recommend_best_model(criterion='accuracy')

    # ========== 步骤8: 生成报告 ==========
    print("[步骤8] 生成报告...")
    print("-" * 70)

    hub.save_comparison_table()
    hub.plot_comparison_charts()
    hub.generate_all_reports()
    hub.generate_final_report()

    print("\n" + "="*70)
    print("✅ 快速示例运行完成！")
    print("="*70)
    print(f"\n输出目录: {hub.output_dir}")
    print("\n📚 下一步：")
    print("   1. 查看 INTEGRATION_SUMMARY.md 了解如何使用完整功能")
    print("   2. 查看 data_integrity_checklist.md 学习最佳实践")
    print("   3. 修改此脚本以添加更多模型")
    print()


if __name__ == '__main__':
    main()
