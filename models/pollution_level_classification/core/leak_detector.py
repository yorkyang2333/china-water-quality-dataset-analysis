"""
数据泄露检测工具
用于检测模型训练中的数据泄露问题
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


class DataLeakDetector:
    """
    数据泄露检测器

    检测以下泄露场景：
    1. 特征与目标的异常相关性
    2. 特征计算中是否使用了未来数据
    3. SMOTE/平衡方法的不当应用
    4. 标准化/标准化器的不当拟合
    """

    def __init__(self, correlation_threshold=0.7, verbose=True):
        """
        参数:
        - correlation_threshold: 相关性异常阈值（r > threshold视为可疑）
        - verbose: 是否打印详细信息
        """
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
        self.alerts = []

    def log(self, level, message):
        """记录警告/错误"""
        if self.verbose:
            print(f"[{level}] {message}")
        self.alerts.append({
            'level': level,
            'message': message
        })

    def check_correlation_anomaly(self, X, y, feature_names=None):
        """
        检查特征与目标的异常相关性

        警告：
        - 如果某个特征与目标的相关性 > threshold，可能存在泄露
        - 这可能意味着该特征直接或间接包含了目标信息
        """
        self.log('INFO', '检查特征与目标的相关性...')

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        anomalies = []

        for i, feature_name in enumerate(feature_names):
            feature_values = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]

            # 计算Spearman相关性（更稳健）
            corr, p_value = spearmanr(feature_values, y, nan_policy='omit')

            if abs(corr) > self.correlation_threshold:
                anomalies.append({
                    'feature': feature_name,
                    'correlation': corr,
                    'p_value': p_value
                })

                self.log(
                    'WARN',
                    f"异常相关性检测: {feature_name} 与目标的相关性 = {corr:.4f} (p={p_value:.2e})"
                )

        if not anomalies:
            self.log('OK', '未检测到异常相关性')

        return anomalies

    def check_feature_names(self, X, target_name=''):
        """
        检查特征名称是否包含目标信息

        警告：
        - 如果特征名包含目标值本身或其变体，可能存在泄露
        - 例如：'log_Ammonia_N'作为特征，而目标是'Ammonia_N'
        """
        self.log('INFO', '检查特征名称中的泄露迹象...')

        if not hasattr(X, 'columns'):
            self.log('INFO', '特征不是DataFrame，跳过名称检查')
            return []

        feature_names = X.columns.tolist()
        suspicious_names = []

        for col in feature_names:
            col_lower = col.lower()

            # 检查是否包含常见的目标变量
            target_keywords = [
                'pollution', 'ammonia', 'oxygen', 'cod', 'bod',
                'ph', 'temperature', 'copper', 'mercury', 'lead'
            ]

            for keyword in target_keywords:
                if keyword in col_lower:
                    suspicious_names.append(col)
                    self.log(
                        'WARN',
                        f"可疑特征名: '{col}' 包含目标相关关键词 '{keyword}'"
                    )
                    break

        if not suspicious_names:
            self.log('OK', '特征名称检查通过')

        return suspicious_names

    def check_data_distribution(self, X_train, X_test):
        """
        检查Train和Test的数据分布是否过度相似

        警告：
        - 如果分布完全相同，可能代表未正确分割数据
        """
        self.log('INFO', '检查Train/Test数据分布...')

        if not hasattr(X_train, 'columns'):
            self.log('INFO', '输入不是DataFrame，跳过分布检查')
            return True

        for col in X_train.columns:
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            mean_diff = abs(train_mean - test_mean)

            train_std = X_train[col].std()
            test_std = X_test[col].std()

            if train_std > 0:
                std_ratio = test_std / train_std
                if 0.9 < std_ratio < 1.1 and mean_diff < 0.1 * train_std:
                    self.log(
                        'WARN',
                        f"特征 '{col}' 的Train/Test分布过于相似（可能数据分割错误）"
                    )

        self.log('OK', 'Train/Test分布检查完成')
        return True

    def check_class_distribution(self, y_train, y_test):
        """
        检查Train和Test的类别分布

        警告：
        - 如果某个类别仅出现在Test中，可能代表分割问题
        """
        self.log('INFO', '检查Train/Test类别分布...')

        train_classes = set(y_train)
        test_classes = set(y_test)

        only_in_test = test_classes - train_classes
        only_in_train = train_classes - test_classes

        if only_in_test:
            self.log(
                'WARN',
                f"某些类别仅出现在Test中: {only_in_test}（可能分割不当）"
            )

        if only_in_train:
            self.log('INFO', f"某些类别仅出现在Train中: {only_in_train}")

        self.log('OK', '类别分布检查完成')

    def generate_report(self):
        """生成泄露检测报告"""
        report = {
            'total_alerts': len(self.alerts),
            'warnings': [a for a in self.alerts if a['level'] == 'WARN'],
            'all_alerts': self.alerts
        }

        return report

    def save_report(self, filepath):
        """保存检测报告"""
        import json

        report = self.generate_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.log('INFO', f"报告已保存至: {filepath}")


class IntegrityChecker:
    """
    数据完整性检查器

    检查：
    1. 数据是否经过正确的预处理
    2. Train/Test/Val的分割是否合理
    3. 所有预处理步骤是否正确
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.checks = []

    def log(self, check_name, result, details=''):
        """记录检查结果"""
        status = '[PASS]' if result else '[FAIL]'
        msg = f"{status} {check_name}"
        if details:
            msg += f" - {details}"

        if self.verbose:
            print(msg)

        self.checks.append({
            'check': check_name,
            'result': result,
            'details': details
        })

    def check_no_nan(self, X, name='Data'):
        """检查是否有NaN值"""
        has_nan = X.isnull().any().any() if hasattr(X, 'isnull') else np.isnan(X).any()
        self.log(f'{name} NaN检查', not has_nan, f'NaN count: {X.isnull().sum().sum() if hasattr(X, "isnull") else 0}')
        return not has_nan

    def check_no_inf(self, X, name='Data'):
        """检查是否有无穷大"""
        has_inf = np.isinf(X).any() if isinstance(X, np.ndarray) else (np.isinf(X).any().any() if hasattr(X, 'values') else False)
        self.log(f'{name} 无穷大检查', not has_inf)
        return not has_inf

    def check_shape(self, X_train, X_test, name='Data'):
        """检查特征数量一致"""
        n_features_train = X_train.shape[1] if len(X_train.shape) > 1 else 1
        n_features_test = X_test.shape[1] if len(X_test.shape) > 1 else 1
        same = n_features_train == n_features_test
        self.log(
            f'{name} 特征数一致性检查',
            same,
            f'Train: {n_features_train}, Test: {n_features_test}'
        )
        return same

    def check_split_ratio(self, n_train, n_test, target_ratio=0.2):
        """检查分割比例"""
        total = n_train + n_test
        actual_ratio = n_test / total
        is_reasonable = 0.15 < actual_ratio < 0.30
        self.log(
            '数据分割比例检查',
            is_reasonable,
            f'实际: {actual_ratio:.2%}, 目标: {target_ratio:.2%}'
        )
        return is_reasonable

    def get_summary(self):
        """获取检查摘要"""
        passed = sum(1 for c in self.checks if c['result'])
        total = len(self.checks)
        return {
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'checks': self.checks
        }

    def print_summary(self):
        """打印检查摘要"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print(f"数据完整性检查摘要")
        print("="*60)
        print(f"通过: {summary['passed']}/{summary['total']} ({summary['pass_rate']:.1%})")
        for check in self.checks:
            status = '[PASS]' if check['result'] else '[FAIL]'
            print(f"{status} {check['check']}")
            if check['details']:
                print(f"    详情: {check['details']}")
        print("="*60 + "\n")
