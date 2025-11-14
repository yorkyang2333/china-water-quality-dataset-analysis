"""
模型对比脚本

功能：
1. 从已保存的评估结果加载模型性能
2. 进行详细的对比分析
3. 生成对比报告和可视化
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd

# 添加core模块到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.utils import get_project_root
from model_hub.comparison import ModelComparison


def load_evaluation_results(results_dir):
    """
    加载评估结果

    参数:
    - results_dir: 包含评估结果的目录

    返回:
    - 结果字典 {model_id: metrics_dict}
    """
    results = {}

    if not os.path.exists(results_dir):
        print(f"[警告] 目录不存在: {results_dir}")
        return results

    # 查找所有模型评估报告
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)

        if not os.path.isdir(model_path):
            continue

        # 查找evaluation_report.json
        report_file = os.path.join(model_path, 'evaluation_report.json')

        if os.path.exists(report_file):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    metrics = report.get('metrics', {})
                    results[model_dir] = metrics
                    print(f"[加载] {model_dir} - {len(metrics)} 个指标")
            except Exception as e:
                print(f"[错误] 加载 {model_dir} 失败: {e}")

    return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print("水质污染分类模型 - 对比分析脚本")
    print("="*70 + "\n")

    # ========== 步骤1: 加载评估结果 ==========
    print("[步骤1] 加载评估结果...")
    print("-" * 70)

    project_root = get_project_root()
    results_dir = os.path.join(
        project_root, 'Outputs', 'Pollution_Classification', 'models'
    )

    results = load_evaluation_results(results_dir)

    if not results:
        print(f"[警告] 未找到评估结果，请先运行 run_all_models.py\n")
        return

    print(f"[成功] 加载了 {len(results)} 个模型的评估结果\n")

    # ========== 步骤2: 创建对比分析器 ==========
    print("[步骤2] 创建对比分析器...")
    print("-" * 70)

    comparator = ModelComparison(results)
    print("[成功] 对比分析器已创建\n")

    # ========== 步骤3: 打印汇总统计 ==========
    print("[步骤3] 汇总统计分析...")
    print("-" * 70)

    comparator.print_summary()

    # ========== 步骤4: 打印排名 ==========
    print("[步骤4] 模型排名分析...")
    print("-" * 70)

    for metric in ['accuracy', 'f1_weighted', 'f1_macro']:
        comparator.print_ranking(metric)

    # ========== 步骤5: 性能差距分析 ==========
    print("[步骤5] 性能差距分析...")
    print("-" * 70)

    for metric in ['accuracy', 'f1_weighted', 'f1_macro']:
        comparator.print_performance_gaps(metric)

    # ========== 步骤6: 统计显著性检验 ==========
    print("[步骤6] 统计显著性检验...")
    print("-" * 70)

    for metric in ['accuracy', 'f1_weighted']:
        comparator.test_statistical_significance(metric, method='anova')

    # ========== 步骤7: 生成可视化 ==========
    print("[步骤7] 生成可视化...")
    print("-" * 70)

    output_dir = os.path.join(
        project_root, 'Outputs', 'Pollution_Classification', 'Comparison'
    )
    os.makedirs(output_dir, exist_ok=True)

    # 热力图
    heatmap_path = os.path.join(output_dir, 'comparison_heatmap.png')
    comparator.plot_comparison_heatmap(output_path=heatmap_path)

    # 雷达图
    radar_path = os.path.join(output_dir, 'comparison_radar.png')
    try:
        comparator.plot_radar_chart(
            output_path=radar_path,
            metrics=['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro']
        )
    except Exception as e:
        print(f"[警告] 雷达图生成失败: {e}")

    # 箱线图
    boxplot_path = os.path.join(output_dir, 'accuracy_boxplot.png')
    comparator.plot_metrics_boxplot('accuracy', output_path=boxplot_path)

    print(f"[成功] 可视化已保存到: {output_dir}\n")

    # ========== 步骤8: 生成对比报告 ==========
    print("[步骤8] 生成对比报告...")
    print("-" * 70)

    report_content = f"""
{'='*70}
水质污染分类模型 - 完整对比报告
{'='*70}

一、模型统计
{'='*70}
总模型数: {len(results)}

模型列表:
"""
    for model_id in results.keys():
        report_content += f"  - {model_id}\n"

    # 汇总统计
    report_content += f"\n二、汇总统计\n{'='*70}\n"
    summary = comparator.get_summary_statistics()
    for metric, stats in summary.items():
        report_content += f"\n{metric}:\n"
        report_content += f"  平均值: {stats['mean']:.4f}\n"
        report_content += f"  标准差: {stats['std']:.4f}\n"
        report_content += f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]\n"

    # 排名
    report_content += f"\n三、模型排名（按准确率）\n{'='*70}\n"
    ranking = comparator.get_ranking('accuracy')
    if ranking is not None:
        for idx, (model_id, row) in enumerate(ranking.iterrows(), 1):
            report_content += f"{idx}. {model_id}: {row['accuracy']:.4f}\n"

    report_content += f"\n{'='*70}\n生成时间: {pd.Timestamp.now()}\n"

    # 保存报告
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"[保存] 对比报告: {report_path}\n")

    # ========== 步骤9: 保存对比表 ==========
    print("[步骤9] 保存对比表...")
    print("-" * 70)

    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    comparator.df.to_csv(csv_path)
    print(f"[保存] 对比表: {csv_path}\n")

    # 打印最终总结
    print("="*70)
    print("对比分析完成！")
    print("="*70)
    print(f"\n输出目录: {output_dir}\n")


if __name__ == '__main__':
    main()
