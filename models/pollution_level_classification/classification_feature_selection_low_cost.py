"""
低成本传感器特征筛选和降维实验

目标：通过特征筛选和降维方法改进低成本传感器污染等级分类性能

基础特征（6个）：
- Turbidity_NTU (浊度)
- pH (pH值)
- Conductivity_uS_cm (电导率)
- Water_Temperature_C (水温)
- Latitude (纬度)
- Longitude (经度)

测试方法：
1. 相关性分析（Pearson/Spearman）
2. 互信息（Mutual Information）
3. 树模型特征重要性
4. PCA降维
5. 特征组合 + PCA

对比基准：原始6特征（准确率38.62%）
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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
    while current_dir != current_dir.parent:
        if (current_dir / 'preprocessed_data_feature_eng.csv').exists():
            return current_dir
        current_dir = current_dir.parent
    return Path(__file__).resolve().parent.parent.parent


# 低成本传感器特征
LOW_COST_FEATURES = [
    'Turbidity_NTU',
    'pH',
    'Conductivity_uS_cm',
    'Water_Temperature_C',
    'Latitude',
    'Longitude'
]

TARGET = 'Pollution_Level'
STATION_COL = 'Monitoring_Station'


def load_data(data_path):
    """加载数据"""
    print(f"\n[加载数据] {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[数据维度] {df.shape}")
    return df


def prepare_data(df, features, target, station_col, test_size=0.2, random_state=42):
    """准备数据（随机分割和站点分割）"""
    # 检查缺失值
    X = df[features].copy()
    y = df[target].copy()
    groups = df[station_col].copy()

    missing_mask = X.isnull().any(axis=1) | y.isnull() | groups.isnull()
    if missing_mask.sum() > 0:
        X = X[~missing_mask]
        y = y[~missing_mask]
        groups = groups[~missing_mask]

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 随机分割
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # 站点分割
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))
    X_train_station = X.iloc[train_idx]
    X_test_station = X.iloc[test_idx]
    y_train_station = y_encoded[train_idx]
    y_test_station = y_encoded[test_idx]

    return {
        'random': (X_train_rand, X_test_rand, y_train_rand, y_test_rand),
        'station': (X_train_station, X_test_station, y_train_station, y_test_station),
        'label_encoder': le
    }


def analyze_correlation(X_train, y_train, feature_names):
    """相关性分析"""
    print("\n" + "="*70)
    print("[方法1] 相关性分析")
    print("="*70)

    # Pearson相关系数
    from scipy.stats import pearsonr, spearmanr

    pearson_scores = []
    spearman_scores = []

    for i, feat in enumerate(feature_names):
        # Pearson
        p_corr, p_pval = pearsonr(X_train.iloc[:, i], y_train)
        pearson_scores.append((feat, abs(p_corr), p_pval))

        # Spearman
        s_corr, s_pval = spearmanr(X_train.iloc[:, i], y_train)
        spearman_scores.append((feat, abs(s_corr), s_pval))

    # 排序
    pearson_scores.sort(key=lambda x: x[1], reverse=True)
    spearman_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n[Pearson相关系数]")
    for feat, corr, pval in pearson_scores:
        print(f"  {feat:<25} |r|={corr:.4f}  (p={pval:.4e})")

    print("\n[Spearman秩相关]")
    for feat, corr, pval in spearman_scores:
        print(f"  {feat:<25} ρ={corr:.4f}  (p={pval:.4e})")

    # 筛选：取Pearson相关系数 > 0.05的特征
    threshold = 0.05
    selected_features = [feat for feat, corr, pval in pearson_scores if corr > threshold]

    print(f"\n[筛选结果] 阈值|r|>{threshold}: {len(selected_features)}个特征")
    print(f"  {selected_features}")

    return selected_features if len(selected_features) > 0 else [pearson_scores[0][0]]


def analyze_mutual_information(X_train, y_train, feature_names):
    """互信息分析"""
    print("\n" + "="*70)
    print("[方法2] 互信息分析")
    print("="*70)

    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    mi_ranking = list(zip(feature_names, mi_scores))
    mi_ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n[互信息分数]")
    for feat, score in mi_ranking:
        print(f"  {feat:<25} MI={score:.4f}")

    # 筛选：取互信息 > 0的特征（实际上会取top k）
    k = max(3, len([s for s in mi_scores if s > 0.01]))  # 至少保留3个
    selected_features = [feat for feat, score in mi_ranking[:k]]

    print(f"\n[筛选结果] Top {k}个特征")
    print(f"  {selected_features}")

    return selected_features


def analyze_tree_importance(X_train, y_train, feature_names):
    """树模型特征重要性"""
    print("\n" + "="*70)
    print("[方法3] 树模型特征重要性")
    print("="*70)

    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    importance_ranking = list(zip(feature_names, importances))
    importance_ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n[特征重要性]")
    cumulative = 0
    for feat, imp in importance_ranking:
        cumulative += imp
        print(f"  {feat:<25} Importance={imp:.4f}  (累计={cumulative:.4f})")

    # 筛选：累计重要性达90%的特征
    threshold = 0.90
    selected_features = []
    cumulative = 0
    for feat, imp in importance_ranking:
        selected_features.append(feat)
        cumulative += imp
        if cumulative >= threshold:
            break

    print(f"\n[筛选结果] 累计重要性≥{threshold}: {len(selected_features)}个特征")
    print(f"  {selected_features}")

    return selected_features


def apply_pca(X_train, X_test, n_components=0.95):
    """应用PCA降维"""
    print("\n" + "="*70)
    print(f"[方法4] PCA降维（保留方差比例≥{n_components}）")
    print("="*70)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"\n[PCA结果]")
    print(f"  原始特征数: {X_train.shape[1]}")
    print(f"  降维后主成分数: {X_train_pca.shape[1]}")
    print(f"  解释方差比例: {pca.explained_variance_ratio_}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_test_pca, X_train_pca.shape[1]


def create_interaction_features(X_train, X_test, feature_names):
    """创建交互特征"""
    print("\n" + "="*70)
    print("[方法5] 特征组合 + PCA")
    print("="*70)

    from sklearn.preprocessing import PolynomialFeatures

    # 创建二阶多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print(f"\n[多项式特征]")
    print(f"  原始特征数: {X_train.shape[1]}")
    print(f"  多项式特征数: {X_train_poly.shape[1]}")

    # 应用PCA降维
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"\n[PCA降维后]")
    print(f"  主成分数: {X_train_pca.shape[1]}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_test_pca, X_train_pca.shape[1]


def train_and_evaluate(X_train, X_test, y_train, y_test, method_name):
    """训练XGBoost并评估"""
    from sklearn.utils.class_weight import compute_class_weight

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    # 训练模型
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Very Poor类（类别4）召回率
    class_4_indices = (y_test == 4)
    if class_4_indices.sum() > 0:
        class_4_recall = recall_score(y_test, y_pred, labels=[4], average='macro')
    else:
        class_4_recall = None

    print(f"\n[{method_name} - 评估结果]")
    print(f"  准确率: {acc:.4f}")
    print(f"  宏平均F1: {f1_macro:.4f}")
    print(f"  加权F1: {f1_weighted:.4f}")
    if class_4_recall is not None:
        print(f"  Very Poor类召回率: {class_4_recall:.4f}")

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'class_4_recall': class_4_recall
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print("低成本传感器特征筛选和降维实验")
    print("="*70)

    # 获取项目根目录
    project_root = get_project_root()
    print(f"\n[项目根目录] {project_root}")

    # 数据路径
    data_path = project_root / 'preprocessed_data_feature_eng.csv'

    # 加载数据
    df = load_data(data_path)

    # 准备数据
    print(f"\n[准备数据]")
    print(f"  特征: {len(LOW_COST_FEATURES)}个")
    print(f"  {LOW_COST_FEATURES}")

    data_splits = prepare_data(df, LOW_COST_FEATURES, TARGET, STATION_COL)

    X_train_rand, X_test_rand, y_train_rand, y_test_rand = data_splits['random']
    X_train_station, X_test_station, y_train_station, y_test_station = data_splits['station']

    print(f"  随机分割: 训练集{len(X_train_rand)}条, 测试集{len(X_test_rand)}条")
    print(f"  站点分割: 训练集{len(X_train_station)}条, 测试集{len(X_test_station)}条")

    # 存储结果
    results = {}

    # ========== 基准：原始6特征 ==========
    print("\n" + "="*70)
    print("[基准] 原始6特征")
    print("="*70)

    results['原始6特征_随机'] = train_and_evaluate(
        X_train_rand, X_test_rand, y_train_rand, y_test_rand, "原始6特征_随机分割"
    )
    results['原始6特征_站点'] = train_and_evaluate(
        X_train_station, X_test_station, y_train_station, y_test_station, "原始6特征_站点分割"
    )

    # ========== 方法1: 相关性筛选 ==========
    selected_corr = analyze_correlation(X_train_rand, y_train_rand, LOW_COST_FEATURES)

    if len(selected_corr) < len(LOW_COST_FEATURES):
        X_train_corr_rand = X_train_rand[selected_corr]
        X_test_corr_rand = X_test_rand[selected_corr]
        X_train_corr_station = X_train_station[selected_corr]
        X_test_corr_station = X_test_station[selected_corr]

        results['相关性筛选_随机'] = train_and_evaluate(
            X_train_corr_rand, X_test_corr_rand, y_train_rand, y_test_rand, f"相关性筛选({len(selected_corr)}特征)_随机分割"
        )
        results['相关性筛选_站点'] = train_and_evaluate(
            X_train_corr_station, X_test_corr_station, y_train_station, y_test_station, f"相关性筛选({len(selected_corr)}特征)_站点分割"
        )
    else:
        print("  [跳过] 相关性筛选未减少特征数")

    # ========== 方法2: 互信息筛选 ==========
    selected_mi = analyze_mutual_information(X_train_rand, y_train_rand, LOW_COST_FEATURES)

    X_train_mi_rand = X_train_rand[selected_mi]
    X_test_mi_rand = X_test_rand[selected_mi]
    X_train_mi_station = X_train_station[selected_mi]
    X_test_mi_station = X_test_station[selected_mi]

    results['互信息筛选_随机'] = train_and_evaluate(
        X_train_mi_rand, X_test_mi_rand, y_train_rand, y_test_rand, f"互信息筛选({len(selected_mi)}特征)_随机分割"
    )
    results['互信息筛选_站点'] = train_and_evaluate(
        X_train_mi_station, X_test_mi_station, y_train_station, y_test_station, f"互信息筛选({len(selected_mi)}特征)_站点分割"
    )

    # ========== 方法3: 树模型重要性筛选 ==========
    selected_tree = analyze_tree_importance(X_train_rand, y_train_rand, LOW_COST_FEATURES)

    X_train_tree_rand = X_train_rand[selected_tree]
    X_test_tree_rand = X_test_rand[selected_tree]
    X_train_tree_station = X_train_station[selected_tree]
    X_test_tree_station = X_test_station[selected_tree]

    results['树重要性筛选_随机'] = train_and_evaluate(
        X_train_tree_rand, X_test_tree_rand, y_train_rand, y_test_rand, f"树重要性筛选({len(selected_tree)}特征)_随机分割"
    )
    results['树重要性筛选_站点'] = train_and_evaluate(
        X_train_tree_station, X_test_tree_station, y_train_station, y_test_station, f"树重要性筛选({len(selected_tree)}特征)_站点分割"
    )

    # ========== 方法4: PCA降维 ==========
    X_train_pca_rand, X_test_pca_rand, n_components_rand = apply_pca(X_train_rand, X_test_rand)
    X_train_pca_station, X_test_pca_station, n_components_station = apply_pca(X_train_station, X_test_station)

    results['PCA降维_随机'] = train_and_evaluate(
        X_train_pca_rand, X_test_pca_rand, y_train_rand, y_test_rand, f"PCA降维({n_components_rand}主成分)_随机分割"
    )
    results['PCA降维_站点'] = train_and_evaluate(
        X_train_pca_station, X_test_pca_station, y_train_station, y_test_station, f"PCA降维({n_components_station}主成分)_站点分割"
    )

    # ========== 方法5: 特征组合 + PCA ==========
    X_train_poly_rand, X_test_poly_rand, n_poly_rand = create_interaction_features(
        X_train_rand, X_test_rand, LOW_COST_FEATURES
    )
    X_train_poly_station, X_test_poly_station, n_poly_station = create_interaction_features(
        X_train_station, X_test_station, LOW_COST_FEATURES
    )

    results['组合特征+PCA_随机'] = train_and_evaluate(
        X_train_poly_rand, X_test_poly_rand, y_train_rand, y_test_rand, f"组合特征+PCA({n_poly_rand}主成分)_随机分割"
    )
    results['组合特征+PCA_站点'] = train_and_evaluate(
        X_train_poly_station, X_test_poly_station, y_train_station, y_test_station, f"组合特征+PCA({n_poly_station}主成分)_站点分割"
    )

    # ========== 汇总对比 ==========
    print("\n" + "="*70)
    print("汇总对比")
    print("="*70)

    print("\n[随机分割]")
    print(f"{'方法':<20} {'准确率':<10} {'宏平均F1':<10} {'Very Poor召回':<15}")
    print("-"*70)
    for key in ['原始6特征_随机', '相关性筛选_随机', '互信息筛选_随机', '树重要性筛选_随机', 'PCA降维_随机', '组合特征+PCA_随机']:
        if key in results:
            r = results[key]
            method_name = key.replace('_随机', '')
            recall_str = f"{r['class_4_recall']:.4f}" if r['class_4_recall'] is not None else "N/A"
            print(f"{method_name:<20} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {recall_str:<15}")

    print("\n[站点分割]")
    print(f"{'方法':<20} {'准确率':<10} {'宏平均F1':<10} {'Very Poor召回':<15}")
    print("-"*70)
    for key in ['原始6特征_站点', '相关性筛选_站点', '互信息筛选_站点', '树重要性筛选_站点', 'PCA降维_站点', '组合特征+PCA_站点']:
        if key in results:
            r = results[key]
            method_name = key.replace('_站点', '')
            recall_str = f"{r['class_4_recall']:.4f}" if r['class_4_recall'] is not None else "N/A"
            print(f"{method_name:<20} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {recall_str:<15}")

    # 保存结果
    output_dir = project_root / 'Outputs' / 'Classification_Low_Cost_Sensors'
    os.makedirs(output_dir, exist_ok=True)

    report_path = output_dir / 'feature_selection_results.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("低成本传感器特征筛选和降维实验结果\n")
        f.write("="*70 + "\n\n")

        f.write("[随机分割]\n")
        f.write(f"{'方法':<20} {'准确率':<10} {'宏平均F1':<10} {'Very Poor召回':<15}\n")
        f.write("-"*70 + "\n")
        for key in ['原始6特征_随机', '相关性筛选_随机', '互信息筛选_随机', '树重要性筛选_随机', 'PCA降维_随机', '组合特征+PCA_随机']:
            if key in results:
                r = results[key]
                method_name = key.replace('_随机', '')
                recall_str = f"{r['class_4_recall']:.4f}" if r['class_4_recall'] is not None else "N/A"
                f.write(f"{method_name:<20} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {recall_str:<15}\n")

        f.write("\n[站点分割]\n")
        f.write(f"{'方法':<20} {'准确率':<10} {'宏平均F1':<10} {'Very Poor召回':<15}\n")
        f.write("-"*70 + "\n")
        for key in ['原始6特征_站点', '相关性筛选_站点', '互信息筛选_站点', '树重要性筛选_站点', 'PCA降维_站点', '组合特征+PCA_站点']:
            if key in results:
                r = results[key]
                method_name = key.replace('_站点', '')
                recall_str = f"{r['class_4_recall']:.4f}" if r['class_4_recall'] is not None else "N/A"
                f.write(f"{method_name:<20} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {recall_str:<15}\n")

    print(f"\n[结果已保存] {report_path}")
    print("\n[完成] 实验结束\n")


if __name__ == '__main__':
    main()
