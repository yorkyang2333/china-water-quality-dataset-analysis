"""
模型对比分析工具
"""

import pandas as pd
import numpy as np
import zhplot  # 自动配置matplotlib中文字体
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal


class ModelComparison:
    """
    模型对比分析工具

    功能：
    1. 多模型性能对比
    2. 统计显著性检验
    3. 性能可视化
    """

    def __init__(self, results_dict):
        """
        初始化

        参数:
        - results_dict: 模型结果字典 {model_id: metrics_dict}
        """
        self.results = results_dict
        self.df = self._build_dataframe()

    def _build_dataframe(self):
        """构建结果数据框"""
        df = pd.DataFrame(self.results).T
        return df

    def get_summary_statistics(self):
        """获取汇总统计"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        summary = {}
        for col in numeric_cols:
            summary[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median()
            }

        return summary

    def print_summary(self):
        """打印汇总统计"""
        summary = self.get_summary_statistics()

        print("\n" + "="*70)
        print("模型性能汇总统计")
        print("="*70)

        for metric, stats in summary.items():
            print(f"\n{metric}:")
            print(f"  平均值: {stats['mean']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
            print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  中位数: {stats['median']:.4f}")

        print("="*70 + "\n")

    def get_ranking(self, metric='accuracy'):
        """获取模型排名"""
        if metric not in self.df.columns:
            print(f"[错误] 指标不存在: {metric}")
            return None

        ranking = self.df[[metric]].sort_values(by=metric, ascending=False)
        ranking['rank'] = range(1, len(ranking) + 1)

        return ranking

    def print_ranking(self, metric='accuracy'):
        """打印排名"""
        ranking = self.get_ranking(metric)

        if ranking is None:
            return

        print(f"\n{metric} 排名:")
        print("="*50)

        for idx, (model_id, row) in enumerate(ranking.iterrows(), 1):
            print(f"{idx}. {model_id}: {row[metric]:.4f}")

        print("="*50 + "\n")

    def test_statistical_significance(self, metric='accuracy', method='anova'):
        """
        检验统计显著性

        参数:
        - metric: 要检验的指标
        - method: 检验方法 ('anova' 或 'kruskal')

        返回:
        - (统计量, p值) 元组
        """
        if metric not in self.df.columns:
            print(f"[错误] 指标不存在: {metric}")
            return None

        values = self.df[[metric]].values.flatten()

        if method == 'anova':
            statistic, p_value = f_oneway(*[values])  # 单组，简化处理
            test_name = 'One-way ANOVA'
        else:
            statistic, p_value = kruskal(values)
            test_name = 'Kruskal-Wallis'

        print(f"\n{test_name} 检验 ({metric}):")
        print(f"  统计量: {statistic:.4f}")
        print(f"  p值: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  结论: 在p<0.05水平上，模型间差异显著 [显著]")
        else:
            print(f"  结论: 在p<0.05水平上，模型间差异不显著 [不显著]")

        return statistic, p_value

    def get_pairwise_differences(self, metric='accuracy'):
        """获取模型间的两两差异"""
        if metric not in self.df.columns:
            return None

        values = self.df[[metric]].values.flatten()
        models = self.df.index.tolist()

        differences = {}
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:
                    diff = values[i] - values[j]
                    differences[f"{model1} vs {model2}"] = diff

        return differences

    def plot_comparison_heatmap(self, output_path=None, figsize=(12, 8)):
        """
        绘制对比热力图

        参数:
        - output_path: 保存路径
        - figsize: 图像大小
        """
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            print("[警告] 无数值数据可绘制")
            return

        # 标准化到[0,1]范围
        normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())

        plt.figure(figsize=figsize)
        sns.heatmap(
            normalized_df, annot=True, fmt='.3f', cmap='RdYlGn',
            cbar_kws={'label': '标准化分数'}
        )
        plt.title('模型性能热力图（标准化）')
        plt.ylabel('模型')
        plt.xlabel('指标')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[保存] 热力图: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_radar_chart(self, output_path=None, metrics=None):
        """
        绘制雷达图

        参数:
        - output_path: 保存路径
        - metrics: 要绘制的指标列表（默认为所有数值列）
        """
        if metrics is None:
            metrics = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # 限制指标数量
        if len(metrics) > 8:
            metrics = metrics[:8]

        numeric_df = self.df[metrics]

        # 标准化
        normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for model_id, row in normalized_df.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_id)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        ax.set_title('模型性能雷达图', y=1.08)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[保存] 雷达图: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_metrics_boxplot(self, metric='accuracy', output_path=None, figsize=(8, 6)):
        """
        绘制箱线图

        参数:
        - metric: 要绘制的指标
        - output_path: 保存路径
        - figsize: 图像大小
        """
        if metric not in self.df.columns:
            print(f"[错误] 指标不存在: {metric}")
            return

        plt.figure(figsize=figsize)
        self.df.boxplot(column=metric)
        plt.title(f'{metric} 分布')
        plt.ylabel(metric.capitalize())
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[保存] 箱线图: {output_path}")
        else:
            plt.show()

        plt.close()

    def get_performance_gaps(self, metric='accuracy'):
        """获取性能差距"""
        if metric not in self.df.columns:
            return None

        values = self.df[[metric]].values.flatten()
        max_val = np.max(values)
        min_val = np.min(values)
        gap = max_val - min_val

        return {
            'best': max_val,
            'worst': min_val,
            'gap': gap,
            'gap_percentage': (gap / max_val) * 100
        }

    def print_performance_gaps(self, metric='accuracy'):
        """打印性能差距"""
        gaps = self.get_performance_gaps(metric)

        if gaps is None:
            return

        print(f"\n{metric} 性能差距分析:")
        print("="*50)
        print(f"最佳: {gaps['best']:.4f}")
        print(f"最差: {gaps['worst']:.4f}")
        print(f"差距: {gaps['gap']:.4f} ({gaps['gap_percentage']:.1f}%)")
        print("="*50 + "\n")
