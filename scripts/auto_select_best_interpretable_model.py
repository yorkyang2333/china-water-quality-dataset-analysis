#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动选择可解释性最佳的模型

功能：
1. 评估所有候选模型的性能和可解释性
2. 根据评分公式自动选择最佳模型
3. 生成详细的选择报告

评分标准：
- 性能权重：0.6（准确率/R^2）
- 可解释性权重：0.4（树模型 + SHAP效率）

作者：Claude Code
日期：2025-11-19
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class ModelSelector:
    """自动模型选择器"""

    def __init__(self, output_dir=None):
        """
        初始化选择器

        Args:
            output_dir: 输出目录（默认为时间戳文件夹）
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "Outputs" / f"SHAP_Analysis_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据路径
        self.data_path = project_root / "preprocessed_data_feature_eng.csv"

        # 分类模型候选
        self.classification_candidates = [
            {
                'name': 'XGBoost',
                'type': 'classification',
                'is_tree_model': True,
                'has_fast_shap': True,  # TreeExplainer
                'script_path': 'models/pollution_level_classification/models/classification_xgboost.py',
                'expected_accuracy': 0.9958
            },
            {
                'name': 'RandomForest',
                'type': 'classification',
                'is_tree_model': True,
                'has_fast_shap': True,
                'script_path': 'models/pollution_level_classification/models/classification_random_forest.py',
                'expected_accuracy': 0.9955
            },
            {
                'name': 'CatBoost',
                'type': 'classification',
                'is_tree_model': True,
                'has_fast_shap': True,
                'script_path': 'models/pollution_level_classification/models/classification_catboost.py',
                'expected_accuracy': 0.9950
            },
            {
                'name': 'LightGBM',
                'type': 'classification',
                'is_tree_model': True,
                'has_fast_shap': True,
                'script_path': 'models/pollution_level_classification/models/classification_lightgbm.py',
                'expected_accuracy': 0.9950
            }
        ]

        # 回归模型候选
        self.regression_candidates = [
            {
                'name': 'XGBoost_Regression_Clean',
                'type': 'regression',
                'is_tree_model': True,
                'has_fast_shap': True,
                'script_path': 'models/indicator_regression/regression_xgboost_clean.py',
                'expected_r2': -0.037,  # 真实性能（无数据泄露）
                'note': '无数据泄露版本，真实性能评估'
            }
        ]

        # 评分权重
        self.performance_weight = 0.6
        self.interpretability_weight = 0.4

    def calculate_interpretability_score(self, model_info):
        """
        计算可解释性得分

        Args:
            model_info: 模型信息字典

        Returns:
            可解释性得分（0-1）
        """
        score = 0.0

        # 树模型 +0.5
        if model_info['is_tree_model']:
            score += 0.5

        # 支持快速SHAP（TreeExplainer）+0.5
        if model_info['has_fast_shap']:
            score += 0.5

        return score

    def calculate_performance_score(self, model_info, actual_performance=None):
        """
        计算性能得分（归一化到0-1）

        Args:
            model_info: 模型信息字典
            actual_performance: 实际测量的性能（如果有）

        Returns:
            性能得分（0-1）
        """
        if model_info['type'] == 'classification':
            # 分类：准确率已经在0-1之间
            perf = actual_performance if actual_performance else model_info['expected_accuracy']
            return perf
        else:
            # 回归：R^2可能为负，需要处理
            perf = actual_performance if actual_performance else model_info['expected_r2']
            # 将R^2映射到0-1，使用sigmoid函数
            # R^2=1 -> score=1, R^2=0 -> score=0.5, R^2<0 -> score<0.5
            if perf >= 0:
                return 0.5 + 0.5 * perf  # 0.5到1
            else:
                # 负值：使用指数衰减
                return 0.5 * np.exp(perf)  # <0.5

    def calculate_total_score(self, model_info, actual_performance=None):
        """
        计算总得分

        Args:
            model_info: 模型信息字典
            actual_performance: 实际性能

        Returns:
            总得分（0-1）
        """
        perf_score = self.calculate_performance_score(model_info, actual_performance)
        interp_score = self.calculate_interpretability_score(model_info)

        total = (self.performance_weight * perf_score +
                self.interpretability_weight * interp_score)

        return total

    def train_and_evaluate_classification(self, model_name):
        """
        训练并评估分类模型（简化版）

        Args:
            model_name: 模型名称

        Returns:
            准确率
        """
        print(f"  [INFO] 训练并评估分类模型: {model_name}")

        # 加载数据
        df = pd.read_csv(self.data_path)

        # 准备特征和目标（基于实际数据集列名）
        feature_columns = [
            'Latitude', 'Longitude',
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm',
            'Turbidity_NTU', 'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
            'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
            'Coliform_Count_CFU_100mL',
            'Total_Heavy_Metals_ug_L', 'COD_BOD_Ratio', 'NP_Ratio'
        ]

        X = df[feature_columns]
        y = df['Pollution_Level']

        # 处理新特征中的无穷大值和NaN
        for col in ['COD_BOD_Ratio', 'NP_Ratio']:
            if col in X.columns:
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 根据模型名称训练
        if model_name == 'XGBoost':
            from xgboost import XGBClassifier
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            model = XGBClassifier(random_state=42, n_jobs=-1)
            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)

            # 保存模型
            model_path = self.output_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump((model, le, feature_columns), f)

        elif model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)

            # 保存模型
            model_path = self.output_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump((model, le, feature_columns), f)

        elif model_name == 'CatBoost':
            from catboost import CatBoostClassifier
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            model = CatBoostClassifier(random_state=42, verbose=0)
            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)

            # 保存模型
            model_path = self.output_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump((model, le, feature_columns), f)

        elif model_name == 'LightGBM':
            from lightgbm import LGBMClassifier
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)

            # 保存模型
            model_path = self.output_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump((model, le, feature_columns), f)

        else:
            raise ValueError(f"未知模型: {model_name}")

        print(f"  [OK] {model_name} 训练完成，准确率: {accuracy:.4f}")
        return accuracy

    def train_and_evaluate_regression(self):
        """
        训练并评估回归模型（清洁版XGBoost）

        Returns:
            平均R^2得分
        """
        print("  [INFO] 训练并评估回归模型: XGBoost_Regression_Clean")

        # 加载数据
        df = pd.read_csv(self.data_path)

        # 定义无泄露特征（基础数值指标）
        base_features = [
            'Latitude', 'Longitude',
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm',
            'Turbidity_NTU', 'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
            'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
            'Coliform_Count_CFU_100mL'
        ]

        # 清洁特征（无泄露）
        clean_features = [
            'Total_Heavy_Metals_ug_L',  # 重金属总和
            'COD_BOD_Ratio',             # COD/BOD比值
            'NP_Ratio',                  # N/P比值
        ]

        # 目标指标（物理/生物过程指标，难以预测）
        target_indicators = [
            'Dissolved_Oxygen_mg_L',
            'Ammonia_N_mg_L',
            'pH'
        ]

        r2_scores = []

        for target in target_indicators:
            # 准备特征（排除目标指标本身）
            feature_cols = [f for f in base_features if f != target] + clean_features

            X = df[feature_cols]
            y = df[target]

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 训练XGBoost回归
            from xgboost import XGBRegressor

            model = XGBRegressor(random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)

            print(f"    {target}: R^2 = {r2:.4f}")

        avg_r2 = np.mean(r2_scores)
        print(f"  [OK] 平均R^2 = {avg_r2:.4f}")

        # 保存最后一个模型作为示例
        model_path = self.output_dir / "XGBoost_Regression_Clean_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump((model, feature_cols, target), f)

        return avg_r2

    def select_best_models(self):
        """
        选择最佳模型

        Returns:
            (best_classification, best_regression, report)
        """
        print("\n" + "="*80)
        print("开始自动模型选择")
        print("="*80)

        results = {
            'classification': [],
            'regression': []
        }

        # 评估分类模型
        print("\n[STEP 1/2] 评估分类模型...")
        for model_info in self.classification_candidates:
            print(f"\n处理模型: {model_info['name']}")

            # 训练并评估
            accuracy = self.train_and_evaluate_classification(model_info['name'])

            # 计算得分
            perf_score = self.calculate_performance_score(model_info, accuracy)
            interp_score = self.calculate_interpretability_score(model_info)
            total_score = self.calculate_total_score(model_info, accuracy)

            results['classification'].append({
                'model_info': model_info,
                'actual_accuracy': accuracy,
                'performance_score': perf_score,
                'interpretability_score': interp_score,
                'total_score': total_score
            })

        # 评估回归模型
        print("\n[STEP 2/2] 评估回归模型...")
        for model_info in self.regression_candidates:
            print(f"\n处理模型: {model_info['name']}")

            # 训练并评估
            avg_r2 = self.train_and_evaluate_regression()

            # 计算得分
            perf_score = self.calculate_performance_score(model_info, avg_r2)
            interp_score = self.calculate_interpretability_score(model_info)
            total_score = self.calculate_total_score(model_info, avg_r2)

            results['regression'].append({
                'model_info': model_info,
                'actual_r2': avg_r2,
                'performance_score': perf_score,
                'interpretability_score': interp_score,
                'total_score': total_score
            })

        # 排序并选择最佳模型
        results['classification'].sort(key=lambda x: x['total_score'], reverse=True)
        results['regression'].sort(key=lambda x: x['total_score'], reverse=True)

        best_classification = results['classification'][0]
        best_regression = results['regression'][0]

        # 生成报告
        report = self.generate_selection_report(results, best_classification, best_regression)

        # 保存报告
        report_path = self.output_dir / "00_selection_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存JSON元数据
        metadata = {
            'classification': {
                'best_model': best_classification['model_info']['name'],
                'accuracy': float(best_classification['actual_accuracy']),
                'total_score': float(best_classification['total_score']),
                'model_path': str(self.output_dir / f"{best_classification['model_info']['name']}_model.pkl")
            },
            'regression': {
                'best_model': best_regression['model_info']['name'],
                'avg_r2': float(best_regression['actual_r2']),
                'total_score': float(best_regression['total_score']),
                'model_path': str(self.output_dir / "XGBoost_Regression_Clean_model.pkl")
            },
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }

        json_path = self.output_dir / "selected_models.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] 报告已保存到: {report_path}")
        print(f"[OK] 元数据已保存到: {json_path}")

        return best_classification, best_regression, report

    def generate_selection_report(self, results, best_classification, best_regression):
        """
        生成选择报告

        Args:
            results: 所有结果
            best_classification: 最佳分类模型
            best_regression: 最佳回归模型

        Returns:
            报告文本
        """
        lines = []
        lines.append("="*80)
        lines.append("自动模型可解释性选择报告")
        lines.append("="*80)
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"输出目录: {self.output_dir}")

        # 评分标准
        lines.append("\n" + "-"*80)
        lines.append("评分标准")
        lines.append("-"*80)
        lines.append(f"性能权重: {self.performance_weight}")
        lines.append(f"可解释性权重: {self.interpretability_weight}")
        lines.append("\n可解释性得分规则:")
        lines.append("  - 树模型: +0.5")
        lines.append("  - 支持TreeExplainer（快速SHAP）: +0.5")
        lines.append("\n性能得分规则:")
        lines.append("  - 分类: 准确率（0-1）")
        lines.append("  - 回归: R^2映射（R^2>=0: 0.5+0.5*R^2, R^2<0: 0.5*exp(R^2)）")

        # 分类模型结果
        lines.append("\n" + "="*80)
        lines.append("分类模型评估结果")
        lines.append("="*80)
        lines.append(f"\n{'排名':<6} {'模型名称':<20} {'准确率':<10} {'性能得分':<10} {'可解释性':<10} {'总得分':<10}")
        lines.append("-"*80)

        for i, result in enumerate(results['classification'], 1):
            model_name = result['model_info']['name']
            accuracy = result['actual_accuracy']
            perf = result['performance_score']
            interp = result['interpretability_score']
            total = result['total_score']

            marker = " [最佳]" if i == 1 else ""
            lines.append(f"{i:<6} {model_name:<20} {accuracy:<10.4f} {perf:<10.4f} {interp:<10.4f} {total:<10.4f}{marker}")

        # 回归模型结果
        lines.append("\n" + "="*80)
        lines.append("回归模型评估结果")
        lines.append("="*80)
        lines.append(f"\n{'排名':<6} {'模型名称':<30} {'平均R^2':<10} {'性能得分':<10} {'可解释性':<10} {'总得分':<10}")
        lines.append("-"*80)

        for i, result in enumerate(results['regression'], 1):
            model_name = result['model_info']['name']
            r2 = result['actual_r2']
            perf = result['performance_score']
            interp = result['interpretability_score']
            total = result['total_score']
            note = result['model_info'].get('note', '')

            marker = " [最佳]" if i == 1 else ""
            lines.append(f"{i:<6} {model_name:<30} {r2:<10.4f} {perf:<10.4f} {interp:<10.4f} {total:<10.4f}{marker}")
            if note:
                lines.append(f"       备注: {note}")

        # 最终选择
        lines.append("\n" + "="*80)
        lines.append("最终选择结果")
        lines.append("="*80)

        lines.append("\n[分类任务] 最佳模型:")
        lines.append(f"  模型名称: {best_classification['model_info']['name']}")
        lines.append(f"  准确率: {best_classification['actual_accuracy']:.4f}")
        lines.append(f"  总得分: {best_classification['total_score']:.4f}")
        lines.append(f"  模型类型: 树模型")
        lines.append(f"  SHAP支持: TreeExplainer（快速）")
        lines.append(f"  保存路径: {best_classification['model_info']['name']}_model.pkl")

        lines.append("\n[回归任务] 最佳模型:")
        lines.append(f"  模型名称: {best_regression['model_info']['name']}")
        lines.append(f"  平均R^2: {best_regression['actual_r2']:.4f}")
        lines.append(f"  总得分: {best_regression['total_score']:.4f}")
        lines.append(f"  模型类型: 树模型")
        lines.append(f"  SHAP支持: TreeExplainer（快速）")
        lines.append(f"  备注: {best_regression['model_info'].get('note', '无')}")
        lines.append(f"  保存路径: XGBoost_Regression_Clean_model.pkl")

        # 下一步
        lines.append("\n" + "="*80)
        lines.append("下一步操作")
        lines.append("="*80)
        lines.append("\n请运行以下脚本生成完整的SHAP可解释性分析:")
        lines.append(f"  python scripts/generate_shap_analysis.py --input {self.output_dir / 'selected_models.json'}")

        lines.append("\n" + "="*80)

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='自动选择可解释性最佳的模型')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认为时间戳文件夹）')

    args = parser.parse_args()

    # 创建选择器
    selector = ModelSelector(output_dir=args.output_dir)

    # 选择最佳模型
    best_clf, best_reg, report = selector.select_best_models()

    # 打印报告
    print("\n" + report)

    print("\n[DONE] 模型选择完成！")


if __name__ == '__main__':
    main()
