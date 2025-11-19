#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成完整的SHAP可解释性分析报告

功能：
1. 全局特征重要性（Summary Plot + Bar Plot）
2. 特征交互分析（Dependence Plot + Interaction）
3. 样本级别解释（Force Plot + Waterfall）
4. 决策路径可视化（Decision Plot）

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

import zhplot  # 自动配置matplotlib中文字体
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免tkinter错误
import shap

from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """SHAP完整分析器"""

    def __init__(self, metadata_path):
        """
        初始化分析器

        Args:
            metadata_path: selected_models.json 路径
        """
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.output_dir = Path(self.metadata['output_dir'])
        self.data_path = project_root / "preprocessed_data_feature_eng.csv"

        # 加载数据
        self.df = pd.read_csv(self.data_path)

        print(f"[OK] 元数据加载成功: {metadata_path}")
        print(f"[OK] 输出目录: {self.output_dir}")

    def analyze_classification_model(self):
        """分析分类模型"""
        print("\n" + "="*80)
        print("分析分类模型")
        print("="*80)

        # 加载模型
        model_name = self.metadata['classification']['best_model']
        model_path = Path(self.metadata['classification']['model_path'])

        print(f"\n[INFO] 加载模型: {model_name}")
        print(f"[INFO] 路径: {model_path}")

        with open(model_path, 'rb') as f:
            model, label_encoder, feature_columns = pickle.load(f)

        # 准备数据
        X = self.df[feature_columns]
        y = self.df['Pollution_Level']

        # 分割数据（使用相同的随机种子）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 创建输出目录
        output_subdir = self.output_dir / f"Classification_{model_name}"
        output_subdir.mkdir(exist_ok=True)

        print(f"[INFO] 计算SHAP值...")

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 对于多分类，shap_values可能是列表或3D数组
        print(f"[INFO] SHAP值形状: {shap_values.shape if hasattr(shap_values, 'shape') else f'列表长度{len(shap_values)}'}")

        # 检测是否为3D数组（新版本LightGBM多分类）
        if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            # (samples, features, classes) -> 使用所有类别的平均值进行可视化
            print(f"[INFO] 检测到多分类模型（3D数组），共{shap_values.shape[2]}个类别")
            # 对于summary_plot等函数，我们使用第一个类别或平均值
            # 这里我们保持原始的3D数组，在各个函数中分别处理
        elif isinstance(shap_values, list):
            # 旧版本：列表形式（每个类别一个数组）
            print(f"[INFO] 检测到多分类模型（列表形式），共{len(shap_values)}个类别")

        print(f"[OK] SHAP值计算完成")

        # 1. 全局特征重要性
        print(f"\n[STEP 1/4] 生成全局特征重要性...")
        self._generate_global_importance(
            shap_values, X_test, output_subdir, feature_columns, is_classification=True
        )

        # 2. 特征交互分析
        print(f"\n[STEP 2/4] 生成特征交互分析...")
        self._generate_interaction_analysis(
            shap_values, X_test, output_subdir, feature_columns, is_classification=True
        )

        # 3. 样本级别解释
        print(f"\n[STEP 3/4] 生成样本级别解释...")
        self._generate_sample_explanations(
            shap_values, X_test, y_test, label_encoder, output_subdir, feature_columns, is_classification=True
        )

        # 4. 决策路径可视化
        print(f"\n[STEP 4/4] 生成决策路径可视化...")
        self._generate_decision_plot(
            explainer.expected_value, shap_values, X_test, y_test, label_encoder,
            output_subdir, feature_columns, is_classification=True
        )

        # 生成报告
        print(f"\n[INFO] 生成分析报告...")
        report = self._generate_classification_report(
            model_name, X_test, feature_columns, shap_values, label_encoder
        )

        report_path = output_subdir / "SHAP_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[OK] 分类模型分析完成！")
        print(f"[OK] 输出目录: {output_subdir}")

    def analyze_regression_model(self):
        """分析回归模型"""
        print("\n" + "="*80)
        print("分析回归模型")
        print("="*80)

        # 加载模型
        model_name = self.metadata['regression']['best_model']
        model_path = Path(self.metadata['regression']['model_path'])

        print(f"\n[INFO] 加载模型: {model_name}")
        print(f"[INFO] 路径: {model_path}")

        with open(model_path, 'rb') as f:
            model, feature_columns, target_name = pickle.load(f)

        # 准备数据
        X = self.df[feature_columns]
        y = self.df[target_name]

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 创建输出目录
        output_subdir = self.output_dir / f"Regression_{model_name}"
        output_subdir.mkdir(exist_ok=True)

        print(f"[INFO] 计算SHAP值...")

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        print(f"[OK] SHAP值计算完成")

        # 1. 全局特征重要性
        print(f"\n[STEP 1/4] 生成全局特征重要性...")
        self._generate_global_importance(
            shap_values, X_test, output_subdir, feature_columns, is_classification=False
        )

        # 2. 特征交互分析
        print(f"\n[STEP 2/4] 生成特征交互分析...")
        self._generate_interaction_analysis(
            shap_values, X_test, output_subdir, feature_columns, is_classification=False
        )

        # 3. 样本级别解释
        print(f"\n[STEP 3/4] 生成样本级别解释...")
        self._generate_sample_explanations_regression(
            shap_values, X_test, y_test, explainer.expected_value, output_subdir, feature_columns
        )

        # 4. 决策路径可视化
        print(f"\n[STEP 4/4] 生成决策路径可视化...")
        self._generate_decision_plot_regression(
            explainer.expected_value, shap_values, X_test, y_test,
            output_subdir, feature_columns
        )

        # 生成报告
        print(f"\n[INFO] 生成分析报告...")
        report = self._generate_regression_report(
            model_name, target_name, X_test, feature_columns, shap_values
        )

        report_path = output_subdir / "SHAP_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[OK] 回归模型分析完成！")
        print(f"[OK] 输出目录: {output_subdir}")

    def _generate_global_importance(self, shap_values, X_test, output_dir, feature_columns, is_classification):
        """生成全局特征重要性图"""
        # 对于3D数组（新版本多分类），取第一个类别或所有类别的平均
        if is_classification and hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            # 使用所有类别的绝对值平均
            shap_for_plot = np.abs(shap_values).mean(axis=2)
        elif is_classification and isinstance(shap_values, list):
            # 旧版本列表形式，使用第一个类别
            shap_for_plot = shap_values[0]
        else:
            shap_for_plot = shap_values

        # Summary Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_for_plot, X_test, feature_names=feature_columns, show=False)
        plt.title('全局特征重要性 - SHAP Summary Plot', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "01_global_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] 01_global_summary.png")

        # Bar Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_for_plot, X_test, feature_names=feature_columns, plot_type="bar", show=False)
        plt.title('全局特征重要性 - SHAP Bar Plot', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "02_global_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] 02_global_bar.png")

    def _generate_interaction_analysis(self, shap_values, X_test, output_dir, feature_columns, is_classification):
        """生成特征交互分析"""
        # 确保feature_columns是列表
        if hasattr(X_test, 'columns'):
            feature_list = X_test.columns.tolist()
        else:
            feature_list = list(feature_columns)

        # 计算平均绝对SHAP值以确定Top特征
        if is_classification and isinstance(shap_values, list):
            # 旧版本SHAP：列表形式
            mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
        elif is_classification and len(shap_values.shape) == 3:
            # 新版本SHAP多分类：(samples, features, classes)
            # 先对samples求平均，再对classes求平均
            mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
        else:
            # 回归或二分类：(samples, features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        top_indices_array = np.argsort(mean_abs_shap)[::-1][:5]
        top_indices = [idx.item() if hasattr(idx, 'item') else int(idx) for idx in top_indices_array]
        top_features = [feature_list[i] for i in top_indices]

        # 准备用于dependence plot的SHAP值（需要2D数组）
        if is_classification and hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_for_dep = np.abs(shap_values).mean(axis=2)
        elif is_classification and isinstance(shap_values, list):
            shap_for_dep = shap_values[0]
        else:
            shap_for_dep = shap_values

        # Dependence Plot for Top 5 features
        for i, (feat_idx, feat_name) in enumerate(zip(top_indices, top_features)):
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feat_idx, shap_for_dep, X_test, feature_names=feature_list, show=False)
            plt.title(f'特征依赖图 - {feat_name}', fontsize=14, pad=15)
            plt.tight_layout()
            safe_name = feat_name.replace('/', '_').replace('\\', '_')
            plt.savefig(output_dir / f"03_dependence_{safe_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] 03_dependence_{safe_name}.png")

        # Interaction Plot for Top 3 pairs
        print("  [INFO] 生成特征交互图（可能需要较长时间）...")
        for i in range(min(3, len(top_indices))):
            for j in range(i+1, min(3, len(top_indices))):
                feat_i = top_features[i]
                feat_j = top_features[j]

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.dependence_plot(top_indices[i], shap_for_dep, X_test,
                                   feature_names=feature_list,
                                   interaction_index=top_indices[j],
                                   show=False)
                plt.title(f'特征交互 - {feat_i} vs {feat_j}', fontsize=14, pad=15)
                plt.tight_layout()
                safe_i = feat_i.replace('/', '_').replace('\\', '_')
                safe_j = feat_j.replace('/', '_').replace('\\', '_')
                plt.savefig(output_dir / f"04_interaction_{safe_i}_vs_{safe_j}.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  [OK] 04_interaction_{safe_i}_vs_{safe_j}.png")

    def _generate_sample_explanations(self, shap_values, X_test, y_test, label_encoder, output_dir, feature_columns, is_classification):
        """生成样本级别解释（分类）"""
        # 获取所有类别
        classes = label_encoder.classes_

        # 对每个类别选择3个代表样本
        for class_idx, class_name in enumerate(classes):
            # 找到该类别的样本
            class_mask = (y_test == class_name).values
            class_indices = np.where(class_mask)[0]

            if len(class_indices) < 3:
                n_samples = len(class_indices)
            else:
                n_samples = 3

            if n_samples == 0:
                continue

            # 随机选择样本
            selected_indices = np.random.choice(class_indices, n_samples, replace=False)

            for sample_idx in selected_indices:
                # 获取该样本的SHAP值
                if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                    # 3D数组：(samples, features, classes)
                    # 取该类别的SHAP值
                    sample_shap = shap_values[sample_idx, :, class_idx]
                    expected_value = 0  # TreeExplainer的基准值
                elif isinstance(shap_values, list):
                    # 列表形式
                    sample_shap = shap_values[class_idx][sample_idx]
                    expected_value = 0
                else:
                    # 2D数组
                    sample_shap = shap_values[sample_idx]
                    expected_value = 0

                # Waterfall Plot
                fig, ax = plt.subplots(figsize=(14, 6))

                # 创建Explanation对象
                if hasattr(X_test, 'columns'):
                    feature_list = X_test.columns.tolist()
                else:
                    feature_list = list(feature_columns)

                explanation = shap.Explanation(
                    values=sample_shap,
                    base_values=expected_value,
                    data=X_test.iloc[sample_idx].values,
                    feature_names=feature_list
                )

                shap.plots.waterfall(explanation, show=False)
                plt.title(f'样本解释 - {class_name} (样本 {sample_idx})', fontsize=14, pad=15)
                plt.tight_layout()
                plt.savefig(output_dir / f"05_waterfall_{class_name}_sample{sample_idx}.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  [OK] 05_waterfall_{class_name}_sample{sample_idx}.png")

    def _generate_sample_explanations_regression(self, shap_values, X_test, y_test, expected_value, output_dir, feature_columns):
        """生成样本级别解释（回归）"""
        # 按预测值分位数选择样本
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        quantiles = [0.1, 0.5, 0.9]  # 低、中、高

        for q in quantiles:
            # 找到接近该分位数的样本
            target_value = np.quantile(y_test_array, q)
            closest_idx = np.argmin(np.abs(y_test_array - target_value))

            # Waterfall Plot
            fig, ax = plt.subplots(figsize=(14, 6))

            explanation = shap.Explanation(
                values=shap_values[closest_idx],
                base_values=expected_value,
                data=X_test.iloc[closest_idx].values,
                feature_names=feature_columns
            )

            shap.plots.waterfall(explanation, show=False)
            label = f"低值" if q == 0.1 else (f"中值" if q == 0.5 else f"高值")
            plt.title(f'样本解释 - {label} (实际值={y_test_array[closest_idx]:.2f})', fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(output_dir / f"05_waterfall_{label}_sample{closest_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] 05_waterfall_{label}_sample{closest_idx}.png")

    def _generate_decision_plot(self, expected_value, shap_values, X_test, y_test, label_encoder, output_dir, feature_columns, is_classification):
        """生成决策路径可视化（分类）"""
        # 选择100个样本
        n_samples = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

        # 准备feature list
        if hasattr(X_test, 'columns'):
            feature_list = X_test.columns.tolist()
        else:
            feature_list = list(feature_columns)

        fig, ax = plt.subplots(figsize=(14, 10))

        # 处理3D SHAP值（多分类）
        if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            # 使用所有类别的平均绝对值
            shap_subset = np.abs(shap_values[sample_indices]).mean(axis=2)
            exp_val = 0
        elif isinstance(shap_values, list):
            # 列表形式：使用第一个类别
            shap_subset = shap_values[0][sample_indices]
            exp_val = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            shap_subset = shap_values[sample_indices]
            exp_val = expected_value

        shap.decision_plot(exp_val, shap_subset, X_test.iloc[sample_indices], feature_names=feature_list, show=False)
        plt.title('决策路径可视化', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "06_decision_plot_all.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] 06_decision_plot_all.png")

    def _generate_decision_plot_regression(self, expected_value, shap_values, X_test, y_test, output_dir, feature_columns):
        """生成决策路径可视化（回归）"""
        # 选择100个样本
        n_samples = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

        fig, ax = plt.subplots(figsize=(14, 10))

        shap.decision_plot(expected_value, shap_values[sample_indices], X_test.iloc[sample_indices], feature_names=feature_columns, show=False)
        plt.title('决策路径可视化', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "06_decision_plot_all.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] 06_decision_plot_all.png")

    def _generate_classification_report(self, model_name, X_test, feature_columns, shap_values, label_encoder):
        """生成分类模型分析报告"""
        # 确保feature_columns是列表
        if hasattr(X_test, 'columns'):
            feature_list = X_test.columns.tolist()
        else:
            feature_list = list(feature_columns)

        lines = []
        lines.append("# SHAP可解释性分析报告 - 分类模型")
        lines.append("")
        lines.append(f"**模型名称**: {model_name}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**测试集样本数**: {len(X_test)}")
        lines.append(f"**特征数量**: {len(feature_list)}")
        lines.append(f"**类别数量**: {len(label_encoder.classes_)}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 1. 全局特征重要性")
        lines.append("")
        lines.append("### 1.1 Summary Plot")
        lines.append("![](01_global_summary.png)")
        lines.append("")
        lines.append("**说明**:")
        lines.append("- 纵轴：按重要性排序的特征")
        lines.append("- 横轴：SHAP值（特征对预测的贡献）")
        lines.append("- 颜色：特征值（红色=高，蓝色=低）")
        lines.append("- 点分布：展示每个特征对所有样本的影响")
        lines.append("")

        lines.append("### 1.2 Bar Plot")
        lines.append("![](02_global_bar.png)")
        lines.append("")
        lines.append("**说明**:")
        lines.append("- 展示平均|SHAP值|，即每个特征的平均重要性")
        lines.append("- 柱状图越长，特征越重要")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 2. 特征交互分析")
        lines.append("")
        lines.append("### 2.1 特征依赖图（Dependence Plots）")
        lines.append("")

        # 计算Top 5特征
        if isinstance(shap_values, list):
            mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        top_indices_array = np.argsort(mean_abs_shap)[::-1][:5]
        top_indices = [idx.item() if hasattr(idx, 'item') else int(idx) for idx in top_indices_array]
        top_features = [feature_list[i] for i in top_indices]

        for feat_name in top_features:
            safe_name = feat_name.replace('/', '_').replace('\\', '_')
            lines.append(f"#### {feat_name}")
            lines.append(f"![](03_dependence_{safe_name}.png)")
            lines.append("")

        lines.append("**说明**:")
        lines.append("- 横轴：特征值")
        lines.append("- 纵轴：SHAP值（该特征对预测的贡献）")
        lines.append("- 颜色：交互特征值（自动选择最强交互特征）")
        lines.append("")

        lines.append("### 2.2 特征交互图（Interaction Plots）")
        lines.append("")
        lines.append("展示Top特征之间的交互效应：")
        lines.append("")

        for i in range(min(3, len(top_features))):
            for j in range(i+1, min(3, len(top_features))):
                feat_i = top_features[i]
                feat_j = top_features[j]
                safe_i = feat_i.replace('/', '_').replace('\\', '_')
                safe_j = feat_j.replace('/', '_').replace('\\', '_')
                lines.append(f"#### {feat_i} vs {feat_j}")
                lines.append(f"![](04_interaction_{safe_i}_vs_{safe_j}.png)")
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 3. 样本级别解释")
        lines.append("")
        lines.append("为每个污染等级选择3个代表样本，展示其预测的详细分解：")
        lines.append("")

        for class_name in label_encoder.classes_:
            lines.append(f"### {class_name}")
            lines.append("")
            # 注意：这里只列出一个示例，实际有多个
            lines.append(f"![](05_waterfall_{class_name}_sample*.png)")
            lines.append("")

        lines.append("**说明**:")
        lines.append("- 从基线值（expected value）出发")
        lines.append("- 每个特征向上推（红色）或向下拉（蓝色）预测值")
        lines.append("- 最终到达实际预测值")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 4. 决策路径可视化")
        lines.append("")
        lines.append("![](06_decision_plot_all.png)")
        lines.append("")
        lines.append("**说明**:")
        lines.append("- 展示100个样本的决策路径")
        lines.append("- 每条线代表一个样本")
        lines.append("- 从基线值开始，逐个特征累积SHAP值")
        lines.append("- 最终到达预测值")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 5. 关键发现")
        lines.append("")
        lines.append("### Top 5 最重要特征：")
        lines.append("")
        for i, feat_name in enumerate(top_features, 1):
            lines.append(f"{i}. **{feat_name}**")
        lines.append("")

        lines.append("### 模型解释性评估：")
        lines.append("")
        lines.append(f"- **模型类型**: {model_name}（树模型）")
        lines.append("- **SHAP支持**: TreeExplainer（高效）")
        lines.append("- **特征数量**: 适中，易于解释")
        lines.append("- **交互效应**: 已识别主要交互特征对")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 6. 使用建议")
        lines.append("")
        lines.append("1. **特征工程**: 关注Top特征，考虑进一步优化")
        lines.append("2. **模型优化**: 分析低重要性特征，考虑剔除以简化模型")
        lines.append("3. **业务应用**: 使用样本解释向利益相关者解释具体预测")
        lines.append("4. **监控重点**: 重点监控Top特征的数据质量")
        lines.append("")

        return "\n".join(lines)

    def _generate_regression_report(self, model_name, target_name, X_test, feature_columns, shap_values):
        """生成回归模型分析报告"""
        # 确保feature_columns是列表
        if hasattr(X_test, 'columns'):
            feature_list = X_test.columns.tolist()
        else:
            feature_list = list(feature_columns)

        lines = []
        lines.append("# SHAP可解释性分析报告 - 回归模型")
        lines.append("")
        lines.append(f"**模型名称**: {model_name}")
        lines.append(f"**预测目标**: {target_name}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**测试集样本数**: {len(X_test)}")
        lines.append(f"**特征数量**: {len(feature_list)}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 1. 全局特征重要性")
        lines.append("")
        lines.append("### 1.1 Summary Plot")
        lines.append("![](01_global_summary.png)")
        lines.append("")
        lines.append("**说明**:")
        lines.append("- 纵轴：按重要性排序的特征")
        lines.append("- 横轴：SHAP值（特征对预测值的影响）")
        lines.append("- 颜色：特征值（红色=高，蓝色=低）")
        lines.append("")

        lines.append("### 1.2 Bar Plot")
        lines.append("![](02_global_bar.png)")
        lines.append("")

        # 计算Top 5特征
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices_array = np.argsort(mean_abs_shap)[::-1][:5]
        top_indices = [idx.item() if hasattr(idx, 'item') else int(idx) for idx in top_indices_array]
        top_features = [feature_list[i] for i in top_indices]

        lines.append("---")
        lines.append("")
        lines.append("## 2. 特征交互分析")
        lines.append("")
        lines.append("### 2.1 特征依赖图")
        lines.append("")

        for feat_name in top_features:
            safe_name = feat_name.replace('/', '_').replace('\\', '_')
            lines.append(f"#### {feat_name}")
            lines.append(f"![](03_dependence_{safe_name}.png)")
            lines.append("")

        lines.append("### 2.2 特征交互图")
        lines.append("")

        for i in range(min(3, len(top_features))):
            for j in range(i+1, min(3, len(top_features))):
                feat_i = top_features[i]
                feat_j = top_features[j]
                safe_i = feat_i.replace('/', '_').replace('\\', '_')
                safe_j = feat_j.replace('/', '_').replace('\\', '_')
                lines.append(f"#### {feat_i} vs {feat_j}")
                lines.append(f"![](04_interaction_{safe_i}_vs_{safe_j}.png)")
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 3. 样本级别解释")
        lines.append("")
        lines.append("按预测值分位数选择代表样本：")
        lines.append("")
        lines.append("### 低值样本")
        lines.append("![](05_waterfall_低值_sample*.png)")
        lines.append("")
        lines.append("### 中值样本")
        lines.append("![](05_waterfall_中值_sample*.png)")
        lines.append("")
        lines.append("### 高值样本")
        lines.append("![](05_waterfall_高值_sample*.png)")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 4. 决策路径可视化")
        lines.append("")
        lines.append("![](06_decision_plot_all.png)")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 5. 关键发现")
        lines.append("")
        lines.append("### Top 5 最重要特征：")
        lines.append("")
        for i, feat_name in enumerate(top_features, 1):
            lines.append(f"{i}. **{feat_name}**")
        lines.append("")

        lines.append("### 注意事项：")
        lines.append("")
        lines.append("⚠️ **数据泄露警告**: 本模型使用清洁版特征（无数据泄露），性能较低但反映真实预测能力。")
        lines.append("")
        lines.append("**适用场景**: 真实性能评估、论文发表、模型对比")
        lines.append("")

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='生成完整的SHAP可解释性分析')
    parser.add_argument('--input', type=str, required=True,
                       help='selected_models.json 路径')

    args = parser.parse_args()

    # 创建分析器
    analyzer = SHAPAnalyzer(args.input)

    # 分析分类模型
    analyzer.analyze_classification_model()

    # 分析回归模型
    analyzer.analyze_regression_model()

    print("\n" + "="*80)
    print("[DONE] SHAP分析完成！")
    print("="*80)
    print(f"\n输出目录: {analyzer.output_dir}")


if __name__ == '__main__':
    main()
