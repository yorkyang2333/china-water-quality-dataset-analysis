"""
XGBoost水质指标回归预测 - 无泄露版本 v2.0
XGBoost Regression for Water Quality Indicators (No Data Leakage)

修改记录（2025-11-25）：
- 扩展到7个目标指标（COD、BOD、TN、TP、DO、NH3-N、pH）
- 为每个目标指标实现严格的特征排除逻辑
- 移除Organic_acid_proxy特征（COD-BOD组合泄露）
- 根据审稿意见重新评估真实R²性能

作者: Claude Code
版本: v2.0
日期: 2025-11-25
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

# 导入无泄露特征工程函数
from feature_engineering_advanced import engineer_clean_features, get_clean_features

print("\n" + "="*80)
print("  XGBoost回归 - 无泄露版本 v2.0 (7个目标指标)")
print("  XGBoost Regression (No Data Leakage) - v2.0")
print("="*80)

# 创建输出目录
OUTPUT_DIR = "../../Outputs/Regression_XGBoost_NoLeakage_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载数据
print("\n[STEP 1] 加载数据...")
data = pd.read_csv("../../preprocessed_data.csv")
print(f"[OK] 数据加载完成: {data.shape}")

# 2. 生成无泄露特征（仅3个：DO饱和度、碱度代理、离子强度代理）
data, clean_features = engineer_clean_features(data)
print(f"[INFO] 无泄露高级特征: {clean_features}")

# 3. 定义目标指标（全部7个）
TARGET_INDICATORS = {
    'COD_mg_L': {
        'name': '化学需氧量',
        'exclude': ['BOD_mg_L'],  # 排除BOD（高度相关，r=0.82）
        'reason': 'BOD与COD高度相关（r=0.82），存在泄露风险'
    },
    'BOD_mg_L': {
        'name': '生化需氧量',
        'exclude': ['COD_mg_L'],  # 排除COD（高度相关）
        'reason': 'COD与BOD高度相关，存在泄露风险'
    },
    'Total_Nitrogen_mg_L': {
        'name': '总氮',
        'exclude': ['Ammonia_N_mg_L', 'Nitrate_mg_L', 'Nitrite_mg_L'],  # 排除氮子成分
        'reason': 'TN = 氨氮 + 硝酸盐 + 硝态氮 + 有机氮，子成分存在泄露'
    },
    'Total_Phosphorus_mg_L': {
        'name': '总磷',
        'exclude': [],  # 数据集无磷子成分，无需排除
        'reason': '无磷子成分指标，无泄露风险'
    },
    'Dissolved_Oxygen_mg_L': {
        'name': '溶解氧',
        'exclude': [],  # DO独立于其他指标
        'reason': '主要受水温和大气交换控制，无化学泄露'
    },
    'Ammonia_N_mg_L': {
        'name': '氨氮',
        'exclude': ['Total_Nitrogen_mg_L', 'Nitrate_mg_L', 'Nitrite_mg_L'],  # 排除TN和其他氮成分
        'reason': '氨氮是TN的子成分，避免循环泄露'
    },
    'pH': {
        'name': 'pH值',
        'exclude': [],  # pH独立
        'reason': 'pH由缓冲系统控制，无直接化学泄露'
    }
}

print(f"\n[STEP 2] 目标指标: {len(TARGET_INDICATORS)} 个")
for i, (ind, info) in enumerate(TARGET_INDICATORS.items(), 1):
    exclude_str = f"排除{len(info['exclude'])}个" if info['exclude'] else "无排除"
    print(f"  {i}. {info['name']} ({ind}): {exclude_str}")

# 4. 定义严格的特征工程函数
def engineer_features_strict(data, target_indicator, exclude_features):
    """
    严格的特征工程（无泄露版本）

    参数：
        data: 输入数据框
        target_indicator: 目标指标名称
        exclude_features: 需要排除的特征列表（防止泄露）

    返回：
        df: 处理后的数据框
        feature_cols: 最终特征列表
    """
    df = data.copy()

    # 所有可用的基础水质指标
    all_base_indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Coliform_Count_CFU_100mL', 'Water_Quality_Index'
    ]

    # 移除目标指标和排除特征
    base_features = [col for col in all_base_indicators
                    if col != target_indicator and col not in exclude_features]

    # 移除缺失值
    required_cols = base_features + [target_indicator, 'Province', 'Date'] + clean_features
    df = df[required_cols].dropna()

    print(f"\n[FEATURE] 特征工程 (目标: {target_indicator})")
    print(f"  有效数据: {len(df)} 条")
    print(f"  排除特征: {exclude_features if exclude_features else '无'}")

    feature_cols = []

    # 1. 基础指标（严格排除后）
    feature_cols.extend(base_features)
    print(f"  基础指标: {len(base_features)} 个 (原14个 - 目标1个 - 排除{len(exclude_features)}个)")

    # 2. 时间特征
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_of_year'] = df['Date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['Day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['Day_of_year'] / 365)
    df['Year'] = df['Date'].dt.year

    time_features = ['Month', 'sin_month', 'cos_month', 'Day_of_year',
                    'sin_day', 'cos_day', 'Year']
    feature_cols.extend(time_features)
    print(f"  时间特征: {len(time_features)} 个")

    # 3. 空间特征
    le_province = LabelEncoder()
    df['Province_encoded'] = le_province.fit_transform(df['Province'])
    feature_cols.append('Province_encoded')
    print(f"  空间特征: 1 个 (省份编码)")

    # 4. 无泄露高级特征（仅3个：DO饱和度、碱度、离子强度）
    for feature in clean_features:
        if feature in df.columns:
            feature_cols.append(feature)

    print(f"  无泄露高级特征: {len([f for f in clean_features if f in df.columns])} 个")
    print(f"\n[OK] 总特征数: {len(feature_cols)}")
    print(f"    = {len(base_features)} 基础 + {len(time_features)} 时间 + 1 空间 + 3 高级")

    return df, feature_cols


# 5. 训练和评估函数
def train_and_evaluate(data, target_indicator, exclude_features, target_name):
    """训练XGBoost并评估（无泄露版本）"""
    print(f"\n{'='*80}")
    print(f"  目标指标: {target_name} ({target_indicator})")
    print(f"{'='*80}")

    # 特征工程
    df, feature_cols = engineer_features_strict(data, target_indicator, exclude_features)

    # 准备X和y
    X = df[feature_cols].values
    y = df[target_indicator].values

    print(f"\n[TRAIN] 准备训练数据...")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  y mean: {y.mean():.2f}, std: {y.std():.2f}")

    # 划分数据集（80/20）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n[SPLIT] 数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    # XGBoost参数
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    # 训练模型
    print(f"\n[MODEL] 训练XGBoost模型...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\n[RESULTS] 模型性能:")
    print(f"  训练集 R²: {train_r2:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}")
    print(f"  测试集 MAE: {test_mae:.4f}")
    print(f"  测试集 RMSE: {test_rmse:.4f}")

    # 判断成功/失败
    if test_r2 >= 0.5:
        status = "[成功]"
    elif test_r2 >= 0:
        status = "[中等]"
    else:
        status = "[失败]"

    print(f"\n  性能评级: {status} (R² = {test_r2:.3f})")

    # 保存结果
    results = {
        'target': target_indicator,
        'target_name': target_name,
        'n_features': len(feature_cols),
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'y_mean': y.mean(),
        'y_std': y.std(),
        'exclude_features': exclude_features,
        'status': status.strip('[]')
    }

    # 绘制散点图（仅为成功/中等指标）
    if test_r2 >= 0:
        plot_prediction_scatter(y_test, y_test_pred, target_indicator, target_name, test_r2)

    return results


def plot_prediction_scatter(y_true, y_pred, target_indicator, target_name, r2):
    """绘制预测vs真实值散点图"""
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_true, y_pred, alpha=0.5, s=30, c='steelblue', edgecolors='none')

    # 完美预测线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, label='完美预测线 (y=x)')

    # 拟合线
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    y_sorted = np.sort(y_true)
    plt.plot(y_sorted, p(y_sorted), 'b-', lw=2, alpha=0.7,
             label=f'拟合线 (y={z[0]:.3f}x+{z[1]:.3f})')

    plt.xlabel(f'真实{target_name}值', fontsize=14, fontweight='bold')
    plt.ylabel(f'预测{target_name}值', fontsize=14, fontweight='bold')
    plt.title(f'{target_name}预测性能 (R²={r2:.3f}，无泄露)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存图片
    filename = f"{OUTPUT_DIR}/{target_indicator}_scatter.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  [OK] 散点图已保存: {filename}")
    plt.close()


# 6. 主程序：运行所有7个指标
print("\n" + "="*80)
print("  开始运行7个目标指标的回归实验（无泄露版本）")
print("="*80)

all_results = []

for target_ind, info in TARGET_INDICATORS.items():
    results = train_and_evaluate(
        data=data,
        target_indicator=target_ind,
        exclude_features=info['exclude'],
        target_name=info['name']
    )
    all_results.append(results)


# 7. 汇总结果
print("\n" + "="*80)
print("  汇总报告：7个目标指标的无泄露性能")
print("="*80)

results_df = pd.DataFrame(all_results)

# 保存详细结果表
results_df.to_csv(f"{OUTPUT_DIR}/regression_results_summary.csv", index=False, encoding='utf-8-sig')
print(f"\n[OK] 详细结果已保存: {OUTPUT_DIR}/regression_results_summary.csv")

# 打印汇总表
print("\n表：XGBoost回归模型性能（无泄露版本，7个目标指标）\n")
print(f"{'目标指标':<20} {'中文名称':<10} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'状态':<6}")
print("-" * 70)

for _, row in results_df.iterrows():
    print(f"{row['target']:<20} {row['target_name']:<10} {row['test_r2']:>8.3f} "
          f"{row['test_mae']:>10.2f} {row['test_rmse']:>10.2f} {row['status']:<6}")

# 分类统计
success_count = (results_df['test_r2'] >= 0.5).sum()
medium_count = ((results_df['test_r2'] >= 0) & (results_df['test_r2'] < 0.5)).sum()
fail_count = (results_df['test_r2'] < 0).sum()

print("\n" + "-" * 70)
print(f"成功指标 (R² >= 0.5): {success_count} 个")
print(f"中等指标 (0 <= R² < 0.5): {medium_count} 个")
print(f"失败指标 (R² < 0): {fail_count} 个")

if success_count > 0:
    avg_r2_success = results_df[results_df['test_r2'] >= 0.5]['test_r2'].mean()
    print(f"\n成功指标平均R²: {avg_r2_success:.3f}")

print("\n" + "="*80)
print("  实验完成！")
print(f"  输出目录: {OUTPUT_DIR}")
print("="*80)
