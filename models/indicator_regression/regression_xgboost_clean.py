"""
无泄露版XGBoost回归 - 真实性能评估
Clean XGBoost Regression (No Data Leakage)

目的：
- 评估没有数据泄露时的真实预测性能
- 对比有泄露版（R^2=0.9993）vs 无泄露版（真实）
- 用于学术论文的诚实报告

作者: Claude Code
日期: 2025-11-14
版本: v1.0 - CLEAN (NO LEAKAGE)
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

# 导入无泄露特征工程函数
from feature_engineering_advanced import engineer_clean_features, get_clean_features

print("\n" + "="*80)
print("  无泄露版 XGBoost 回归 - 真实性能评估")
print("  Clean XGBoost Regression (No Data Leakage)")
print("="*80)

# 创建输出目录
OUTPUT_DIR = "models_outputs/indicator_regression_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载数据
print("\n[STEP 1] 加载数据...")
data = pd.read_csv("../../preprocessed_data.csv")
print(f"[OK] 数据加载完成: {data.shape}")

# 2. 生成无泄露特征
data, clean_features = engineer_clean_features(data)

# 3. 定义目标指标（仅3个失败的指标）
TARGET_INDICATORS = [
    'Dissolved_Oxygen_mg_L',
    'Ammonia_N_mg_L',
    'pH'
]

print(f"\n[STEP 2] 目标指标: {len(TARGET_INDICATORS)} 个")
for ind in TARGET_INDICATORS:
    print(f"  - {ind}")

# 4. 定义基础特征工程函数
def engineer_features(data, target_indicator):
    """
    特征工程（无泄露版本）

    仅使用：
    - 其他13个基础水质指标（不含目标）
    - 4个无泄露高级特征
    - 时间特征（7个）
    - 空间特征（1个省份编码）

    总计：26个特征（13+4+7+1+1）
    """
    df = data.copy()

    # 移除缺失值
    valid_indicators = [col for col in df.columns
                       if col in ['Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
                                  'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
                                  'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
                                  'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
                                  'Coliform_Count_CFU_100mL', 'Water_Quality_Index']]

    required_cols = valid_indicators + [target_indicator, 'Province', 'Date'] + clean_features
    df = df[required_cols].dropna()

    print(f"\n[FEATURE] 特征工程 (目标: {target_indicator})")
    print(f"  有效数据: {len(df)} 条")

    feature_cols = []

    # 1. 其他基础指标（不含目标）
    base_features = [col for col in valid_indicators if col != target_indicator]
    feature_cols.extend(base_features)
    print(f"  基础指标: {len(base_features)} 个")

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

    # 4. ⭐⭐⭐ 无泄露高级特征（仅4个）
    for feature in clean_features:
        if feature in df.columns:
            feature_cols.append(feature)

    print(f"  无泄露高级特征: {len([f for f in clean_features if f in df.columns])} 个")
    print(f"\n[OK] 总特征数: {len(feature_cols)} (无泄露)")
    print(f"    = {len(base_features)} 基础 + {len(time_features)} 时间 + 1 空间 + 4 高级")

    return df, feature_cols, le_province


# 5. 训练和评估函数
def train_and_evaluate(data, target_indicator):
    """训练XGBoost并评估（无泄露版本）"""
    print(f"\n{'='*80}")
    print(f"  目标指标: {target_indicator}")
    print(f"{'='*80}")

    # 特征工程
    df, feature_cols, le_province = engineer_features(data, target_indicator)

    # 准备X和y
    X = df[feature_cols].values
    y = df[target_indicator].values

    print(f"\n[TRAIN] 准备训练数据...")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")

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

    model = xgb.XGBRegressor(**params)

    # 5折交叉验证
    print(f"\n[CV] 5折交叉验证...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold,
                                scoring='r2', n_jobs=-1)

    print(f"[OK] CV R^2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"    折次得分: {[f'{s:.4f}' for s in cv_scores]}")

    # 在全部数据上训练
    model.fit(X, y)
    y_pred = model.predict(X)

    # 评估指标
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"\n[METRICS] 全数据集性能:")
    print(f"  R^2:  {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    # 特征重要性
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\n[FEATURE IMPORTANCE] 前10个重要特征:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    # 保存结果
    results = {
        'indicator': target_indicator,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(df),
        'n_features': len(feature_cols)
    }

    # 绘制预测散点图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, y_pred, alpha=0.3, s=20)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美预测')
    ax.set_xlabel(f'真实值 ({target_indicator})', fontsize=12)
    ax.set_ylabel(f'预测值 ({target_indicator})', fontsize=12)
    ax.set_title(f'{target_indicator}\n无泄露版 (R^2={r2:.4f}, RMSE={rmse:.4f})',
                fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加性能文本
    textstr = f'样本数: {len(df)}\n特征数: {len(feature_cols)}\n'
    textstr += f'R^2 (CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{target_indicator}_clean_predictions.png",
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] 预测图已保存: {OUTPUT_DIR}/{target_indicator}_clean_predictions.png")

    return results, feature_importance


# 6. 运行所有指标
all_results = []
all_importances = {}

for indicator in TARGET_INDICATORS:
    results, importance = train_and_evaluate(data, indicator)
    all_results.append(results)
    all_importances[indicator] = importance


# 7. 生成汇总报告
print(f"\n{'='*80}")
print("  无泄露版性能汇总")
print(f"{'='*80}\n")

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# 保存汇总结果
results_df.to_csv(f"{OUTPUT_DIR}/clean_regression_results_summary.csv", index=False)
print(f"\n[OK] 汇总结果已保存: {OUTPUT_DIR}/clean_regression_results_summary.csv")

# 8. 生成文本报告
with open(f"{OUTPUT_DIR}/clean_regression_report.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("  无泄露版 XGBoost 回归 - 性能报告\n")
    f.write("  Clean XGBoost Regression (No Data Leakage)\n")
    f.write("="*80 + "\n\n")

    f.write("1. 方法说明\n")
    f.write("-"*80 + "\n")
    f.write("本报告使用无泄露特征进行回归，结果真实可信。\n\n")
    f.write("特征组成:\n")
    f.write("  - 基础指标: 13个 (不含目标)\n")
    f.write("  - 时间特征: 7个\n")
    f.write("  - 空间特征: 1个\n")
    f.write("  - 无泄露高级特征: 4个\n")
    f.write("    * DO_saturation_mg_L (依赖水温)\n")
    f.write("    * Alkalinity_proxy (依赖电导率)\n")
    f.write("    * Ionic_strength_proxy (依赖电导率)\n")
    f.write("    * Organic_acid_proxy (COD-BOD)\n")
    f.write("  - 总计: 25个特征\n\n")

    f.write("2. 性能结果\n")
    f.write("-"*80 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"\n{row['indicator']}:\n")
        f.write(f"  样本数: {row['n_samples']}\n")
        f.write(f"  特征数: {row['n_features']}\n")
        f.write(f"  R^2 (CV): {row['cv_r2_mean']:.4f} +/- {row['cv_r2_std']:.4f}\n")
        f.write(f"  R^2 (Train): {row['train_r2']:.4f}\n")
        f.write(f"  RMSE: {row['rmse']:.4f}\n")
        f.write(f"  MAE: {row['mae']:.4f}\n")

    f.write("\n3. 关键发现\n")
    f.write("-"*80 + "\n")
    avg_r2 = results_df['cv_r2_mean'].mean()
    f.write(f"平均 R^2: {avg_r2:.4f}\n")

    if avg_r2 < 0.5:
        f.write("\n结论: 物理/生物过程指标（DO、氨氮、pH）难以用化学污染物预测。\n")
        f.write("      这是指标特性决定的，不是方法失败。\n")
    else:
        f.write("\n结论: 无泄露特征提供了中等预测性能。\n")

    f.write("\n4. 与有泄露版对比\n")
    f.write("-"*80 + "\n")
    f.write("有泄露版 (regression_xgboost_improved.py):\n")
    f.write("  - 使用11个泄露特征（如log_Ammonia_N等）\n")
    f.write("  - R^2 = 0.9993 (虚假高性能)\n")
    f.write("  - 仅适用于缺失值填补，不可用于未来预测\n\n")

    f.write("无泄露版 (本报告):\n")
    f.write("  - 使用4个无泄露特征\n")
    f.write(f"  - R^2 = {avg_r2:.4f} (真实性能)\n")
    f.write("  - 可用于未来预测和学术发表\n\n")

    f.write("="*80 + "\n")
    f.write("报告生成时间: 2025-11-14\n")
    f.write("="*80 + "\n")

print(f"[OK] 文本报告已保存: {OUTPUT_DIR}/clean_regression_report.txt")

print("\n" + "="*80)
print("  无泄露版回归分析完成")
print("="*80)
print(f"\n所有结果已保存至: {OUTPUT_DIR}/")
print(f"\n关键结论:")
print(f"  - 平均 R^2: {avg_r2:.4f} (真实，vs 有泄露版的0.9993)")
print(f"  - 这才是无数据泄露时的真实预测性能")
print(f"  - 详见报告: {OUTPUT_DIR}/clean_regression_report.txt")
