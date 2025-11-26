"""
低成本传感器污染等级分类实验

目标：评估仅使用低成本物理传感器预测污染等级的可行性

低成本特征（6个）：
- Turbidity_NTU (浊度 - 光学传感器)
- pH (pH值 - 电极)
- Conductivity_uS_cm (电导率 - 电极)
- Water_Temperature_C (水温 - 热敏电阻)
- Latitude (纬度)
- Longitude (经度)

对比基准：
- 全指标特征集（20个特征）

分割策略：
1. 随机分割（80-20）
2. 按站点分割（Leave-One-Station-Out）
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[错误] XGBoost未安装，请运行: pip install xgboost")
    sys.exit(1)

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).resolve().parent
    # 向上找到包含preprocessed_data_feature_eng.csv的目录
    while current_dir != current_dir.parent:
        if (current_dir / 'preprocessed_data_feature_eng.csv').exists():
            return current_dir
        current_dir = current_dir.parent
    return Path(__file__).resolve().parent.parent.parent


# 低成本传感器特征
LOW_COST_FEATURES = [
    'Turbidity_NTU',           # 浊度（光学传感器）
    'pH',                      # pH值（电极）
    'Conductivity_uS_cm',      # 电导率（电极）
    'Water_Temperature_C',     # 水温（热敏电阻）
    'Latitude',                # 纬度
    'Longitude'                # 经度
]

# 目标变量
TARGET = 'Pollution_Level'

# 站点列
STATION_COL = 'Monitoring_Station'


def load_data(data_path):
    """加载数据"""
    print(f"\n[加载数据] {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[数据维度] {df.shape}")
    print(f"[列名] {list(df.columns)}")

    return df


def check_features(df, features):
    """检查特征是否存在"""
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"\n[警告] 缺失特征: {missing}")
        print("[可用特征]:")
        for col in df.columns:
            if 'Temp' in col or 'temp' in col or 'pH' in col or 'Turbidity' in col:
                print(f"  - {col}")
        return False
    return True


def prepare_data_random_split(df, features, target, test_size=0.2, random_state=42):
    """准备数据 - 随机分割"""
    print(f"\n[准备数据 - 随机分割]")
    print(f"[特征] {len(features)} 个: {features}")

    # 检查缺失值
    X = df[features].copy()
    y = df[target].copy()

    missing_mask = X.isnull().any(axis=1) | y.isnull()
    if missing_mask.sum() > 0:
        print(f"[警告] 发现 {missing_mask.sum()} 行缺失值，将删除")
        X = X[~missing_mask]
        y = y[~missing_mask]

    print(f"[有效样本] {len(X)} 条")
    print(f"[类别分布]")
    print(y.value_counts().sort_index())

    # 随机分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[训练集] {len(X_train)} 条")
    print(f"[测试集] {len(X_test)} 条")

    return X_train, X_test, y_train, y_test


def prepare_data_station_split(df, features, target, station_col, test_size=0.2, random_state=42):
    """准备数据 - 按站点分割"""
    print(f"\n[准备数据 - 按站点分割]")
    print(f"[特征] {len(features)} 个: {features}")

    # 检查站点列
    if station_col not in df.columns:
        print(f"[错误] 站点列 '{station_col}' 不存在")
        print(f"[可用列] {[c for c in df.columns if 'Station' in c or 'station' in c]}")
        return None, None, None, None

    # 检查缺失值
    X = df[features].copy()
    y = df[target].copy()
    groups = df[station_col].copy()

    missing_mask = X.isnull().any(axis=1) | y.isnull() | groups.isnull()
    if missing_mask.sum() > 0:
        print(f"[警告] 发现 {missing_mask.sum()} 行缺失值，将删除")
        X = X[~missing_mask]
        y = y[~missing_mask]
        groups = groups[~missing_mask]

    print(f"[有效样本] {len(X)} 条")
    print(f"[站点数量] {groups.nunique()} 个")

    # 按站点分割
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_stations = groups.iloc[train_idx].nunique()
    test_stations = groups.iloc[test_idx].nunique()

    print(f"[训练集] {len(X_train)} 条 ({train_stations} 个站点)")
    print(f"[测试集] {len(X_test)} 条 ({test_stations} 个站点)")

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, X_test, y_test):
    """训练XGBoost模型"""
    print(f"\n[训练 XGBoost]")

    # 计算类别权重
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder

    # 如果y_train是字符串，编码为数字
    if y_train.dtype == 'object':
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        classes = le.classes_
        print(f"[类别映射] {dict(zip(classes, range(len(classes))))}")
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
        classes = np.unique(y_train)

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.arange(len(classes)), y=y_train_encoded)
    sample_weights = np.array([class_weights[y] for y in y_train_encoded])

    # 训练模型
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )

    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)

    # 评估
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train_encoded, y_train_pred)
    test_acc = accuracy_score(y_test_encoded, y_test_pred)

    print(f"[训练集准确率] {train_acc:.4f}")
    print(f"[测试集准确率] {test_acc:.4f}")

    # 返回编码后的真实值和预测值
    return model, y_test_encoded, y_test_pred


def evaluate_model(y_true, y_pred, split_type):
    """评估模型 - y_true和y_pred都应该是编码后的数字"""
    print(f"\n[评估结果 - {split_type}]")
    print("="*70)

    # 基本指标
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"准确率: {acc:.4f}")
    print(f"宏平均F1: {f1_macro:.4f}")
    print(f"加权F1: {f1_weighted:.4f}")

    # 分类报告 - 使用标签映射
    # 0: Excellent, 1: Good, 2: Moderate, 3: Poor, 4: Very Poor
    label_names = ['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor']
    print(f"\n[分类报告]")
    report = classification_report(y_true, y_pred, target_names=label_names)
    print(report)

    # 混淆矩阵
    print(f"\n[混淆矩阵]")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 极差类(Very Poor = 4)召回率
    class_4_indices = (y_true == 4)
    if class_4_indices.sum() > 0:
        class_4_recall = recall_score(y_true, y_pred, labels=[4], average='macro')
        print(f"\n[极差类(Very Poor)召回率] {class_4_recall:.4f}")
    else:
        class_4_recall = None
        print(f"\n[极差类召回率] 无极差类样本")

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'class_4_recall': class_4_recall,  # 修正为class_4_recall
        'confusion_matrix': cm,
        'classification_report': report
    }


def save_results(results, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存文本报告
    report_path = os.path.join(output_dir, 'low_cost_sensor_results.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("低成本传感器污染等级分类实验结果\n")
        f.write("="*70 + "\n\n")

        for split_type, result in results.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"{split_type}\n")
            f.write(f"{'='*70}\n")
            f.write(f"准确率: {result['accuracy']:.4f}\n")
            f.write(f"宏平均F1: {result['f1_macro']:.4f}\n")
            f.write(f"加权F1: {result['f1_weighted']:.4f}\n")
            if result['class_4_recall'] is not None:
                f.write(f"极差类召回率: {result['class_4_recall']:.4f}\n")
            f.write(f"\n分类报告:\n")
            f.write(result['classification_report'])
            f.write(f"\n混淆矩阵:\n")
            f.write(str(result['confusion_matrix']))
            f.write("\n\n")

    print(f"\n[结果已保存] {report_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("低成本传感器污染等级分类实验")
    print("="*70)

    # 获取项目根目录
    project_root = get_project_root()
    print(f"\n[项目根目录] {project_root}")

    # 数据路径
    data_path = project_root / 'preprocessed_data_feature_eng.csv'

    # 加载数据
    df = load_data(data_path)

    # 检查特征
    if not check_features(df, LOW_COST_FEATURES + [TARGET, STATION_COL]):
        print("\n[错误] 数据集缺少必要的特征")
        return

    # 存储结果
    results = {}

    # ========== 实验1: 随机分割 ==========
    print("\n" + "="*70)
    print("实验1: 随机分割（80-20）")
    print("="*70)

    X_train, X_test, y_train, y_test = prepare_data_random_split(
        df, LOW_COST_FEATURES, TARGET
    )

    if X_train is not None:
        model, y_test_encoded, y_pred = train_xgboost(X_train, y_train, X_test, y_test)
        results['随机分割'] = evaluate_model(y_test_encoded, y_pred, '随机分割')

    # ========== 实验2: 按站点分割 ==========
    print("\n" + "="*70)
    print("实验2: 按站点分割（空间泛化测试）")
    print("="*70)

    X_train, X_test, y_train, y_test = prepare_data_station_split(
        df, LOW_COST_FEATURES, TARGET, STATION_COL
    )

    if X_train is not None:
        model, y_test_encoded, y_pred = train_xgboost(X_train, y_train, X_test, y_test)
        results['按站点分割'] = evaluate_model(y_test_encoded, y_pred, '按站点分割')

    # 保存结果
    output_dir = project_root / 'Outputs' / 'Classification_Low_Cost_Sensors'
    save_results(results, output_dir)

    # 打印汇总
    print("\n" + "="*70)
    print("汇总对比")
    print("="*70)
    print(f"{'分割策略':<15} {'准确率':<10} {'宏平均F1':<10} {'极差类召回率':<15}")
    print("-"*70)
    for split_type, result in results.items():
        class_4_recall_str = f"{result['class_4_recall']:.4f}" if result['class_4_recall'] else "N/A"
        print(f"{split_type:<15} {result['accuracy']:<10.4f} {result['f1_macro']:<10.4f} {class_4_recall_str:<15}")

    print("\n[完成] 实验结束\n")


if __name__ == '__main__':
    main()
