"""
标准评估器
统一的模型评估、可视化和报告生成
"""

import os
import json
import numpy as np
import pandas as pd
import zhplot  # 自动配置matplotlib中文字体
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


class ClassifierEvaluator:
    """
    分类器评估器

    功能：
    1. 计算各种性能指标
    2. 生成混淆矩阵和分类报告
    3. 绘制评估可视化
    4. 生成完整的评估报告
    """

    def __init__(self, model_name, output_dir, label_mapping=None):
        """
        初始化

        参数:
        - model_name: 模型名称
        - output_dir: 输出目录
        - label_mapping: 标签映射 {0: 'Good', 1: 'Fair', ...}
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.label_mapping = label_mapping or {}
        os.makedirs(output_dir, exist_ok=True)

        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}

    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        评估模型

        参数:
        - y_true: 真实标签
        - y_pred: 预测标签
        - y_pred_proba: 预测概率（可选）
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        # 计算基础指标
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        self.metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 尝试计算ROC-AUC（二分类或多分类one-vs-rest）
        try:
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            pass

        print(f"\n[评估] {self.model_name} 评估完成")
        self._print_metrics()

    def _print_metrics(self):
        """打印指标摘要"""
        print("-" * 60)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("-" * 60)

    def get_metrics(self):
        """获取所有指标"""
        return self.metrics.copy()

    def plot_confusion_matrix(self, filename='confusion_matrix.png', figsize=(8, 6)):
        """
        绘制混淆矩阵

        参数:
        - filename: 保存文件名
        - figsize: 图像大小
        """
        if self.y_true is None or self.y_pred is None:
            print("[错误] 未执行评估，无法绘制混淆矩阵")
            return

        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self._get_labels(),
            yticklabels=self._get_labels()
        )
        plt.title(f'{self.model_name} - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[保存] 混淆矩阵: {path}")
        plt.close()

    def plot_metrics_comparison(self, filename='metrics_comparison.png', figsize=(10, 6)):
        """
        绘制指标对比图

        参数:
        - filename: 保存文件名
        - figsize: 图像大小
        """
        if not self.metrics:
            print("[错误] 未评估，无法绘制指标对比")
            return

        metrics_to_plot = {
            k: v for k, v in self.metrics.items()
            if isinstance(v, (int, float)) and k in [
                'accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro'
            ]
        }

        plt.figure(figsize=figsize)
        bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color='steelblue')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.ylim([0, 1.0])
        plt.title(f'{self.model_name} - 性能指标')
        plt.ylabel('分数')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[保存] 指标对比图: {path}")
        plt.close()

    def plot_per_class_metrics(self, filename='per_class_metrics.png', figsize=(12, 6)):
        """
        绘制各类别的性能指标

        参数:
        - filename: 保存文件名
        - figsize: 图像大小
        """
        if self.y_true is None or self.y_pred is None:
            print("[错误] 未评估，无法绘制分类报告")
            return

        # 计算各类别指标
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)

        labels = self._get_labels()
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision_per_class, width, label='精确率', color='steelblue')
        ax.bar(x, recall_per_class, width, label='召回率', color='darkorange')
        ax.bar(x + width, f1_per_class, width, label='F1分数', color='seagreen')

        ax.set_ylabel('分数')
        ax.set_title(f'{self.model_name} - 各类别性能指标')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[保存] 各类别指标图: {path}")
        plt.close()

    def generate_classification_report(self, filename='classification_report.txt'):
        """
        生成分类报告

        参数:
        - filename: 保存文件名
        """
        if self.y_true is None or self.y_pred is None:
            print("[错误] 未评估，无法生成分类报告")
            return

        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self._get_labels(),
            digits=4
        )

        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"模型: {self.model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)

        print(f"[保存] 分类报告: {path}")

    def generate_json_report(self, filename='evaluation_report.json'):
        """
        生成JSON格式的评估报告

        参数:
        - filename: 保存文件名
        """
        report = {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist()
        }

        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[保存] JSON报告: {path}")

    def generate_all_reports(self):
        """生成所有报告和可视化"""
        print(f"\n[生成] 为 {self.model_name} 生成所有报告...")

        self.plot_confusion_matrix()
        self.plot_metrics_comparison()
        self.plot_per_class_metrics()
        self.generate_classification_report()
        self.generate_json_report()

        print(f"[完成] 所有报告生成完毕\n")

    def _get_labels(self):
        """获取标签名称"""
        if not self.y_true is None and len(self.label_mapping) > 0:
            n_classes = len(np.unique(self.y_true))
            return [self.label_mapping.get(i, f'Class_{i}') for i in range(n_classes)]
        elif self.y_true is not None:
            n_classes = len(np.unique(self.y_true))
            return [f'Class_{i}' for i in range(n_classes)]
        else:
            return list(self.label_mapping.values()) if self.label_mapping else []


class CrossValidationEvaluator:
    """
    交叉验证评估器

    用于评估模型在多个折中的性能
    """

    def __init__(self, model_name, output_dir, n_splits=5):
        """
        初始化

        参数:
        - model_name: 模型名称
        - output_dir: 输出目录
        - n_splits: 折数
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.n_splits = n_splits
        self.fold_results = []

    def add_fold_result(self, fold_id, metrics):
        """添加折的结果"""
        metrics['fold_id'] = fold_id
        self.fold_results.append(metrics)

    def get_summary(self):
        """获取汇总统计"""
        if not self.fold_results:
            return {}

        df = pd.DataFrame(self.fold_results)
        summary = {
            'mean': df.drop('fold_id', axis=1).mean().to_dict(),
            'std': df.drop('fold_id', axis=1).std().to_dict(),
            'min': df.drop('fold_id', axis=1).min().to_dict(),
            'max': df.drop('fold_id', axis=1).max().to_dict()
        }

        return summary

    def plot_fold_performance(self, filename='cv_fold_performance.png', figsize=(12, 6)):
        """绘制各折的性能"""
        if not self.fold_results:
            print("[错误] 无交叉验证结果")
            return

        df = pd.DataFrame(self.fold_results)
        summary = self.get_summary()

        metrics = list(summary['mean'].keys())
        means = [summary['mean'][m] for m in metrics]
        stds = [summary['std'][m] for m in metrics]

        plt.figure(figsize=figsize)
        plt.errorbar(metrics, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
        plt.title(f'{self.model_name} - 交叉验证性能')
        plt.ylabel('分数')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[保存] 交叉验证性能图: {path}")
        plt.close()

    def print_summary(self):
        """打印汇总信息"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print(f"交叉验证汇总 ({self.n_splits} 折)")
        print("="*60)

        print("\n平均值:")
        for key, value in summary['mean'].items():
            print(f"  {key}: {value:.4f} ± {summary['std'][key]:.4f}")

        print("\n范围:")
        for key in summary['mean'].keys():
            print(f"  {key}: [{summary['min'][key]:.4f}, {summary['max'][key]:.4f}]")

        print("="*60 + "\n")
