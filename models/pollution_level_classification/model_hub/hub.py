"""
模型中心管理器
负责模型的注册、训练、评估和管理
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

from .registry import ModelRegistry

# 处理相对导入问题
try:
    from ..core import ClassifierEvaluator
except ImportError:
    # 如果相对导入失败，尝试从绝对路径导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.evaluator import ClassifierEvaluator


class ModelHub:
    """
    模型中心管理器

    功能：
    1. 注册和管理多个模型
    2. 批量训练所有模型
    3. 统一评估和对比
    4. 性能报告生成
    5. 最佳模型推荐
    """

    def __init__(self, output_dir=None):
        """
        初始化

        参数:
        - output_dir: 中心输出目录
        """
        self.registry = ModelRegistry()
        self.models = {}
        self.evaluators = {}
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_dir is None:
            try:
                from ..core.utils import get_project_root
            except ImportError:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from core.utils import get_project_root

            output_dir = os.path.join(
                get_project_root(), 'Outputs', 'Pollution_Classification', 'Hub'
            )

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[初始化] ModelHub 已创建，输出目录: {self.output_dir}")

    def register_model(self, model_id, model_instance, category='custom'):
        """
        注册模型

        参数:
        - model_id: 模型ID（唯一标识）
        - model_instance: 模型实例
        - category: 模型类别（traditional/deep_learning/ensemble）
        """
        self.models[model_id] = model_instance
        self.registry.register(model_id, model_instance, category)
        print(f"[注册] {model_id} 已注册 (类别: {category})")

    def get_model(self, model_id):
        """获取模型实例"""
        return self.models.get(model_id)

    def get_all_models(self):
        """获取所有注册的模型"""
        return self.models.copy()

    def train_all(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        批量训练所有模型

        参数:
        - X_train, y_train: 训练数据
        - X_val, y_val: 验证数据（可选）
        - verbose: 是否打印详细信息
        """
        print(f"\n{'='*70}")
        print(f"开始批量训练 {len(self.models)} 个模型")
        print(f"{'='*70}\n")

        trained_count = 0
        failed_count = 0

        for model_id, model in self.models.items():
            print(f"\n[{model_id}] 开始训练...")

            try:
                model.train(X_train, y_train, X_val, y_val)
                trained_count += 1
                print(f"[{model_id}] 训练完成 [OK]")

            except Exception as e:
                failed_count += 1
                print(f"[{model_id}] 训练失败 [ERROR]: {str(e)}")

        print(f"\n{'='*70}")
        print(f"训练总结: {trained_count}/{len(self.models)} 成功，{failed_count} 失败")
        print(f"{'='*70}\n")

    def evaluate_all(self, X_test, y_test, label_mapping=None, verbose=True):
        """
        批量评估所有模型

        参数:
        - X_test, y_test: 测试数据
        - label_mapping: 标签映射
        - verbose: 是否打印详细信息
        """
        print(f"\n{'='*70}")
        print(f"开始批量评估 {len(self.models)} 个模型")
        print(f"{'='*70}\n")

        evaluated_count = 0
        failed_count = 0

        for model_id, model in self.models.items():
            print(f"\n[{model_id}] 评估中...")

            try:
                # 预测
                y_pred = model.predict(X_test)

                # 尝试获取概率
                y_pred_proba = None
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass

                # 创建评估器
                output_dir = os.path.join(
                    self.output_dir, 'models', model_id
                )
                os.makedirs(output_dir, exist_ok=True)

                evaluator = ClassifierEvaluator(
                    model_id, output_dir, label_mapping
                )

                # 评估
                evaluator.evaluate(y_test, y_pred, y_pred_proba)

                # 保存结果
                self.evaluators[model_id] = evaluator
                self.results[model_id] = evaluator.get_metrics()

                print(f"[{model_id}] 评估完成 [OK]")
                evaluated_count += 1

            except Exception as e:
                failed_count += 1
                print(f"[{model_id}] 评估失败 [ERROR]: {str(e)}")

        print(f"\n{'='*70}")
        print(f"评估总结: {evaluated_count}/{len(self.models)} 成功，{failed_count} 失败")
        print(f"{'='*70}\n")

    def get_comparison_table(self):
        """
        获取模型对比表

        返回:
        - DataFrame，包含所有模型的性能指标
        """
        if not self.results:
            print("[警告] 未执行评估，无可用结果")
            return None

        df = pd.DataFrame(self.results).T

        # 保留关键指标
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro']
        available_metrics = [m for m in key_metrics if m in df.columns]

        if available_metrics:
            df = df[available_metrics]

        # 按准确率排序
        if 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)

        return df

    def print_comparison_table(self):
        """打印对比表"""
        df = self.get_comparison_table()

        if df is None:
            return

        print("\n" + "="*70)
        print("模型性能对比")
        print("="*70)
        print(df.to_string())
        print("="*70 + "\n")

    def save_comparison_table(self, filename='model_comparison.csv'):
        """保存对比表"""
        df = self.get_comparison_table()

        if df is None:
            return

        path = os.path.join(self.output_dir, filename)
        df.to_csv(path)
        print(f"[保存] 对比表: {path}")

    def recommend_best_model(self, criterion='accuracy'):
        """
        推荐最佳模型

        参数:
        - criterion: 评估标准（accuracy, f1_weighted, f1_macro等）

        返回:
        - (model_id, score) 元组
        """
        df = self.get_comparison_table()

        if df is None or criterion not in df.columns:
            print(f"[错误] 无法找到评估标准: {criterion}")
            return None, None

        best_model = df[criterion].idxmax()
        best_score = df.loc[best_model, criterion]

        print(f"\n{'='*70}")
        print(f"最佳模型推荐")
        print(f"{'='*70}")
        print(f"模型: {best_model}")
        print(f"标准: {criterion}")
        print(f"得分: {best_score:.4f}")
        print(f"{'='*70}\n")

        return best_model, best_score

    def generate_all_reports(self):
        """为所有模型生成完整报告"""
        print(f"\n{'='*70}")
        print(f"生成所有模型的报告...")
        print(f"{'='*70}\n")

        for model_id, evaluator in self.evaluators.items():
            print(f"[{model_id}] 生成报告...")
            evaluator.generate_all_reports()

        self._generate_hub_summary_report()

    def _generate_hub_summary_report(self):
        """生成Hub总结报告"""
        report = {
            'timestamp': self.timestamp,
            'total_models': len(self.models),
            'evaluated_models': len(self.evaluators),
            'comparison_table': self.get_comparison_table().to_dict() if self.get_comparison_table() is not None else {}
        }

        path = os.path.join(self.output_dir, 'hub_summary.json')

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"[保存] Hub总结报告: {path}")

    def plot_comparison_charts(self):
        """绘制对比图表"""
        import zhplot  # 自动配置matplotlib中文字体
        import matplotlib.pyplot as plt

        df = self.get_comparison_table()

        if df is None or df.empty:
            print("[警告] 无数据可绘制")
            return

        # 选择要绘制的指标
        metrics_to_plot = ['accuracy', 'f1_weighted', 'f1_macro']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]

        if not available_metrics:
            print("[警告] 无可用指标")
            return

        # 创建对比图
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))

        if len(available_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available_metrics):
            df[metric].plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel(metric.capitalize())
            ax.set_title(f'模型 {metric} 对比')
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim([0, 1.0])

        plt.tight_layout()

        path = os.path.join(self.output_dir, 'comparison_charts.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[保存] 对比图表: {path}")
        plt.close()

    def get_model_info_summary(self):
        """获取所有模型的信息摘要"""
        summary = {}

        for model_id, model in self.models.items():
            summary[model_id] = model.get_model_info()

        return summary

    def print_model_info_summary(self):
        """打印模型信息摘要"""
        summary = self.get_model_info_summary()

        print("\n" + "="*70)
        print("注册模型摘要")
        print("="*70)

        for model_id, info in summary.items():
            print(f"\n{model_id}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

        print("="*70 + "\n")

    def generate_final_report(self):
        """生成最终的完整报告"""
        report_content = f"""
{'='*70}
水质污染分类模型中心 - 最终报告
{'='*70}

生成时间: {datetime.now().isoformat()}

1. 模型统计
{'='*70}
总注册模型数: {len(self.models)}
已评估模型数: {len(self.evaluators)}

2. 模型列表
{'='*70}
"""
        for model_id in self.models.keys():
            status = '[已评估]' if model_id in self.evaluators else '[未评估]'
            report_content += f"  - {model_id} {status}\n"

        # 添加对比表
        df = self.get_comparison_table()
        if df is not None:
            report_content += f"\n3. 性能对比表\n{'='*70}\n"
            report_content += df.to_string() + "\n"

        # 最佳模型
        best_model, best_score = self.recommend_best_model('accuracy')
        if best_model:
            report_content += f"\n4. 最佳模型推荐\n{'='*70}\n"
            report_content += f"模型: {best_model}\n"
            report_content += f"准确率: {best_score:.4f}\n"

        report_content += f"\n{'='*70}\n"

        path = os.path.join(self.output_dir, 'final_report.txt')

        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"[保存] 最终报告: {path}")
        return report_content
