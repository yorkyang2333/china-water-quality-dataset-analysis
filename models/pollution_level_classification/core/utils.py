"""
工具函数模块
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def get_project_root():
    """获取项目根目录（从core模块向上4层）"""
    current_file = os.path.abspath(__file__)
    # core/utils.py -> core -> pollution_level_classification -> models -> 项目根
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))


def get_data_path(filename='preprocessed_data_feature_eng.csv'):
    """获取数据文件路径"""
    project_root = get_project_root()
    return os.path.join(project_root, filename)


def get_output_dir(model_name):
    """获取模型输出目录"""
    project_root = get_project_root()
    output_dir = os.path.join(
        project_root, 'Outputs', 'Pollution_Classification', model_name
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def get_class_weights(y, method='balanced'):
    """
    计算类别权重

    参数:
    - y: 标签数组
    - method: 权重计算方法
      - 'balanced': 根据类频率计算
      - 'log': 对数缩放
      - 'sqrt': 平方根缩放

    返回:
    - 类别权重字典
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(method, classes=classes, y=y)

    weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    return weight_dict


def print_class_distribution(y, label_mapping=None):
    """
    打印类别分布

    参数:
    - y: 标签数组
    - label_mapping: 标签映射字典
    """
    unique, counts = np.unique(y, return_counts=True)

    print("\n[类别分布]")
    print("-" * 40)

    for cls, count in zip(unique, counts):
        percentage = count / len(y) * 100
        label = label_mapping.get(cls, f'Class_{cls}') if label_mapping else f'Class_{cls}'
        print(f"  {label}: {count:5d} ({percentage:5.1f}%)")

    print("-" * 40 + "\n")


def train_test_split_by_time(X, y, test_size=0.2):
    """
    按时间顺序分割数据（用于时间序列数据）

    参数:
    - X, y: 数据和标签
    - test_size: 测试集比例

    返回:
    - X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def compute_model_complexity(model):
    """
    计算模型复杂度（参数数量）

    适用于：sklearn模型、PyTorch模型等

    参数:
    - model: 模型对象

    返回:
    - 参数数量（整数）
    """
    n_params = 0

    if hasattr(model, 'n_estimators'):
        # RandomForest, GradientBoosting等
        return model.n_estimators * model.get_params().get('max_depth', 10)

    elif hasattr(model, 'coef_'):
        # 线性模型
        n_params += np.prod(model.coef_.shape)
        if hasattr(model, 'intercept_'):
            n_params += np.prod(model.intercept_.shape)

    elif hasattr(model, 'support_vectors_'):
        # SVM
        n_params = len(model.support_vectors_)

    else:
        # PyTorch模型或其他
        try:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            pass

    return n_params


def get_data_summary(X, y):
    """
    获取数据摘要

    参数:
    - X: 特征矩阵
    - y: 标签

    返回:
    - 摘要字典
    """
    summary = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'classes': sorted(np.unique(y).tolist()),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
    }

    return summary


def print_data_summary(X, y, label_mapping=None):
    """打印数据摘要"""
    summary = get_data_summary(X, y)

    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    print(f"样本数: {summary['n_samples']}")
    print(f"特征数: {summary['n_features']}")
    print(f"类别数: {summary['n_classes']}")
    print("\n类别分布:")

    for cls, count in sorted(summary['class_distribution'].items()):
        percentage = count / summary['n_samples'] * 100
        label = label_mapping.get(cls, f'Class_{cls}') if label_mapping else f'Class_{cls}'
        print(f"  {label}: {count:5d} ({percentage:5.1f}%)")

    print("="*60 + "\n")


class ProgressBar:
    """简单的进度条"""

    def __init__(self, total, prefix='', length=30):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0

    def update(self, amount=1):
        self.current += amount
        self._print_progress()

    def _print_progress(self):
        percent = self.current / self.total
        filled = int(self.length * percent)
        bar = '[' + '=' * filled + '-' * (self.length - filled) + ']'
        print(f'\r{self.prefix} {bar} {percent*100:.1f}%', end='')

        if self.current >= self.total:
            print()


def validate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    验证预测结果

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_pred_proba: 预测概率

    返回:
    - 验证是否通过（布尔值）
    """
    # 检查形状
    if len(y_true) != len(y_pred):
        print("[错误] y_true 和 y_pred 长度不匹配")
        return False

    # 检查标签范围
    if np.min(y_pred) < 0 or np.max(y_pred) >= len(np.unique(y_true)):
        print("[错误] 预测标签超出范围")
        return False

    # 检查概率
    if y_pred_proba is not None:
        if not np.allclose(y_pred_proba.sum(axis=1), 1.0):
            print("[警告] 预测概率行和不为1")

        if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
            print("[警告] 预测概率不在[0,1]范围内")

    return True


def get_timestamp():
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_git_commit_hash():
    """获取当前git提交哈希（如果在git repo中）"""
    import subprocess
    try:
        hash_val = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        return hash_val[:7]
    except:
        return 'unknown'


def log_experiment(model_name, metrics, extra_info=None):
    """
    记录实验结果

    参数:
    - model_name: 模型名称
    - metrics: 性能指标字典
    - extra_info: 额外信息
    """
    from datetime import datetime
    import json

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': metrics,
        'extra_info': extra_info or {}
    }

    project_root = get_project_root()
    log_file = os.path.join(project_root, 'Outputs', 'experiment_log.jsonl')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    print(f"[记录] 实验结果已保存至: {log_file}")
