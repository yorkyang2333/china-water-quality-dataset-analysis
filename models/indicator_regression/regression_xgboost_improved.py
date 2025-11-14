"""
水质指标回归预测 - XGBoost改进版（有数据泄露）
Improved XGBoost Regression with Advanced Features (WITH DATA LEAKAGE)

⚠️⚠️⚠️ 严重警告：本脚本存在数据泄露问题 ⚠️⚠️⚠️

=============================================================================
数据泄露问题说明 (Data Leakage Warning)
=============================================================================

本脚本中的15个高级特征中，11个存在严重数据泄露！

泄露特征列表：
  1. log_Ammonia_N          → 直接对数变换目标 (相关性 0.996)
  2. DO_saturation_percent  → 公式包含DO目标 (相关性 0.92)
  3. DO_deficit_mg_L        → 公式包含DO目标 (相关性 0.91)
  4. DO_BOD_ratio           → 分子包含DO目标
  5. Ammonia_TN_ratio       → 分子包含氨氮目标 (相关性 0.90)
  6. Total_Inorganic_N_mg_L → 求和包含氨氮目标
  7. Nitrate_Ammonia_ratio  → 分母包含氨氮目标
  8. Organic_N_mg_L         → 差值包含氨氮目标
  9. H_concentration        → 直接变换pH目标 (相关性 0.77)
 10. DO_COD_ratio           → 分子包含DO目标

性能对比：
  - 本脚本（有泄露）: R² = 0.9993  ← 虚假的高性能！
  - 无泄露版本:       R² = -0.037  ← 真实性能
  - 性能差距: 1.036 (103倍！)

=============================================================================
适用场景 (Valid Use Cases)
=============================================================================

✅ 可用：同时刻缺失值填补
  场景：传感器故障导致某指标缺失，但其他指标正常
  示例：DO传感器坏了，已知水温、COD等13个指标，填补DO值
  原因：此时可使用目标值的派生特征（如DO_saturation_percent）

❌ 不可用：未来预测
  场景：预测明天的水质指标
  原因：未来的目标值未知，无法计算泄露特征

❌ 不可用：学术论文主要结果
  原因：性能虚假，不能代表真实预测能力

=============================================================================
正确做法 (Correct Approach)
=============================================================================

如需真实预测性能，请使用无泄露版本：
  → models/indicator_regression/regression_xgboost_clean.py
  → 使用4个无泄露特征
  → 真实性能：R² = -0.037

详细分析见：
  → docs/DATA_LEAKAGE_FINAL_REPORT.md

=============================================================================

⭐⭐⭐ 改进版本 - 添加高级特征工程

改进内容：
  - 添加15个物理化学领域知识驱动的特征
  - 针对溶解氧、氨氮、pH三个失败指标进行优化
  - 对比改进前后的性能提升

作者：Claude Code
日期：2025-11-13
更新：2025-11-14 (添加数据泄露警告)
用途：缺失值填补 + 数据泄露案例分析
状态：仅限特定场景使用，不建议作为通用预测方法
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入特征工程模块
from feature_engineering_advanced import engineer_advanced_features, validate_features

sns.set_style("whitegrid")

# ==================== 配置 ====================
OUTPUT_DIR = 'models_outputs/indicator_regression_improved'
PAPER_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

# 只针对3个失败指标
TARGET_INDICATORS = [
    'Dissolved_Oxygen_mg_L',       # 溶解氧
    'Ammonia_N_mg_L',              # 氨氮
    'pH'                           # pH值
]

# 所有水质指标
ALL_INDICATORS = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
    'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
    'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
    'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
    'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
]

# 工程特征
ENGINEERED_FEATURES = [
    'Total_Heavy_Metals_ug_L',
    'COD_BOD_Ratio',
    'NP_Ratio'
]

# ==================== 数据加载 ====================
def load_data():
    """加载数据"""
    print("\n[LOAD] 正在加载数据...")

    data_paths = [
        'preprocessed_data_feature_eng.csv',
        '../../preprocessed_data_feature_eng.csv',
        'preprocess/preprocessed_data_feature_eng.csv',
    ]

    for path in data_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            print(f"[OK] 从 {path} 加载数据")
            print(f"  数据大小: {data.shape[0]} 行, {data.shape[1]} 列")
            return data

    print("[ERROR] 找不到数据文件")
    return None

# ==================== 特征工程 ====================
def engineer_features(data, target_indicator, advanced_features):
    """
    特征工程（改进版 - 包含高级特征）

    参数：
    - data: 原始数据
    - target_indicator: 目标指标
    - advanced_features: 高级特征列表

    返回：
    - df: 处理后的数据
    - feature_cols: 特征列表
    - le_province: 省份编码器
    """
    print(f"\n[FEATURE] 正在为 {target_indicator} 构建特征...")

    df = data.copy()

    # 处理日期
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # 时间特征
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

    # 季节性循环编码（sin/cos）
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['sin_doy'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

    # 空间特征 - 编码省份
    le_province = LabelEncoder()
    df['Province_encoded'] = le_province.fit_transform(df['Province'])

    # 特征选择：所有指标 - 目标指标
    feature_cols = []

    # 1. 其他水质指标
    for indicator in ALL_INDICATORS:
        if indicator != target_indicator and indicator in df.columns:
            feature_cols.append(indicator)

    # 2. 原有工程特征
    for feature in ENGINEERED_FEATURES:
        if feature in df.columns:
            feature_cols.append(feature)

    # 3. ⭐⭐⭐ 新增：高级特征
    for feature in advanced_features:
        if feature in df.columns:
            feature_cols.append(feature)

    # 4. 时间特征
    feature_cols.extend(['Month', 'Quarter', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy', 'IsWeekend'])

    # 5. 空间特征
    feature_cols.extend(['Province_encoded'])

    # 如果有 station_count 特征，也加入
    if 'station_count' in df.columns:
        feature_cols.append('station_count')

    # 移除缺失值
    df = df.dropna(subset=feature_cols + [target_indicator])

    print(f"[OK] 特征数量: {len(feature_cols)} (原始26个 + 新增{len([f for f in advanced_features if f in df.columns])}个)")
    print(f"[OK] 样本数量: {len(df)}")

    return df, feature_cols, le_province

# ==================== 模型训练与评估 ====================
def train_and_evaluate(data, target_indicator, feature_cols, cv_folds=5):
    """
    训练XGBoost回归模型并使用交叉验证评估
    """
    print(f"\n[TRAIN] 正在训练 {target_indicator} 的XGBoost回归模型...")

    X = data[feature_cols].values
    y = data[target_indicator].values

    # 用于分层的省份标签
    provinces = data['Province'].values

    # XGBoost回归模型
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # 5折交叉验证（按省份分层）
    le_cv = LabelEncoder()
    province_labels = le_cv.fit_transform(provinces)

    # 使用分层K折（确保每折都有各省份的样本）
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # 交叉验证评分
    print(f"[CV] 执行{cv_folds}折交叉验证...")
    cv_scores_r2 = cross_val_score(model, X, y, cv=skf.split(X, province_labels),
                                    scoring='r2', n_jobs=-1)
    cv_scores_neg_mse = cross_val_score(model, X, y, cv=skf.split(X, province_labels),
                                         scoring='neg_mean_squared_error', n_jobs=-1)
    cv_scores_neg_mae = cross_val_score(model, X, y, cv=skf.split(X, province_labels),
                                         scoring='neg_mean_absolute_error', n_jobs=-1)

    cv_rmse = np.sqrt(-cv_scores_neg_mse)
    cv_mae = -cv_scores_neg_mae

    print(f"\n[RESULTS] 交叉验证结果:")
    print(f"  R^2 = {cv_scores_r2.mean():.4f} +/- {cv_scores_r2.std():.4f}")
    print(f"  RMSE = {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")
    print(f"  MAE = {cv_mae.mean():.4f} +/- {cv_mae.std():.4f}")

    # 在全部数据上训练最终模型（用于特征重要性分析和绘图）
    print(f"\n[TRAIN] 在全部数据上训练最终模型...")
    model.fit(X, y)
    y_pred = model.predict(X)

    # 计算最终指标
    r2_final = r2_score(y, y_pred)
    rmse_final = np.sqrt(mean_squared_error(y, y_pred))
    mae_final = mean_absolute_error(y, y_pred)

    try:
        mape_final = mean_absolute_percentage_error(y, y_pred)
    except:
        mape_final = np.nan

    print(f"\n[FINAL] 最终模型性能（全数据）:")
    print(f"  R^2 = {r2_final:.4f}")
    print(f"  RMSE = {rmse_final:.4f}")
    print(f"  MAE = {mae_final:.4f}")
    if not np.isnan(mape_final):
        print(f"  MAPE = {mape_final:.4f}")

    results = {
        'target': target_indicator,
        'cv_r2_mean': cv_scores_r2.mean(),
        'cv_r2_std': cv_scores_r2.std(),
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_mae_mean': cv_mae.mean(),
        'cv_mae_std': cv_mae.std(),
        'final_r2': r2_final,
        'final_rmse': rmse_final,
        'final_mae': mae_final,
        'final_mape': mape_final
    }

    return model, results, y_pred

# ==================== 可视化 ====================
def plot_predictions(y_true, y_pred, target_indicator, results):
    """绘制预测vs实际散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 散点图
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='完美预测')
    axes[0].set_xlabel(f'实际值 ({target_indicator})', fontsize=12)
    axes[0].set_ylabel(f'预测值 ({target_indicator})', fontsize=12)
    axes[0].set_title(f'{target_indicator} 预测结果（改进版）\nR² = {results["final_r2"]:.4f}, RMSE = {results["final_rmse"]:.4f}',
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 残差图
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel(f'预测值 ({target_indicator})', fontsize=12)
    axes[1].set_ylabel('残差 (预测值 - 实际值)', fontsize=12)
    axes[1].set_title('残差分析', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/{target_indicator}_predictions_improved.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 预测图已保存: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_cols, target_indicator, top_n=20):
    """绘制特征重要性（显示更多特征）"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 10))
    plt.barh(range(len(feature_importance)), feature_importance['Importance'].values, alpha=0.8)
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'].values)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title(f'{target_indicator} - 特征重要性分析（改进版，Top {top_n}）',
             fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    save_path = f"{OUTPUT_DIR}/{target_indicator}_feature_importance_improved.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 特征重要性图已保存: {save_path}")
    plt.close()

    return feature_importance

# ==================== 主程序 ====================
def main():
    print("\n" + "="*70)
    print("  水质指标回归预测 - XGBoost改进版（高级特征工程）")
    print("  针对3个失败指标：溶解氧、氨氮、pH")
    print("="*70)

    # 加载数据
    data = load_data()
    if data is None:
        return

    # ⭐⭐⭐ 应用高级特征工程
    data, advanced_features = engineer_advanced_features(data)
    data = validate_features(data, advanced_features)

    # 存储所有结果
    all_results = []

    # 对每个目标指标进行预测
    for target_indicator in TARGET_INDICATORS:
        print(f"\n{'='*70}")
        print(f"[TARGET] 目标指标: {target_indicator}")
        print(f"{'='*70}")

        try:
            # 特征工程
            df, feature_cols, le_province = engineer_features(data, target_indicator, advanced_features)

            # 训练和评估
            model, results, y_pred = train_and_evaluate(df, target_indicator, feature_cols, cv_folds=5)

            # 可视化
            y_true = df[target_indicator].values
            plot_predictions(y_true, y_pred, target_indicator, results)
            feature_importance = plot_feature_importance(model, feature_cols, target_indicator, top_n=20)

            # 保存特征重要性到CSV
            importance_path = f"{OUTPUT_DIR}/{target_indicator}_feature_importance_improved.csv"
            feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
            print(f"[OK] 特征重要性已保存: {importance_path}")

            all_results.append(results)

        except Exception as e:
            print(f"[ERROR] 处理 {target_indicator} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成总结报告
    print(f"\n{'='*70}")
    print("  总结报告（改进版）")
    print(f"{'='*70}")

    results_df = pd.DataFrame(all_results)

    # 控制台输出
    print("\n交叉验证结果汇总（改进版）:")
    print(results_df[['target', 'cv_r2_mean', 'cv_r2_std', 'cv_rmse_mean', 'cv_mae_mean']].to_string(index=False))

    # 保存到CSV
    results_path = f"{OUTPUT_DIR}/regression_results_improved_summary.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 结果已保存: {results_path}")

    # 生成报告文件
    report_path = f"{OUTPUT_DIR}/regression_improved_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("水质指标回归预测 - XGBoost改进版\n")
        f.write("="*70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: XGBoost回归 + 高级特征工程 + 5折交叉验证\n")
        f.write(f"目标指标数量: {len(TARGET_INDICATORS)}\n")
        f.write(f"新增特征数量: {len(advanced_features)}\n")
        f.write("="*70 + "\n\n")

        f.write("交叉验证结果汇总:\n")
        f.write("-"*70 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"\n【{row['target']}】\n")
            f.write(f"  R^2 = {row['cv_r2_mean']:.4f} +/- {row['cv_r2_std']:.4f}\n")
            f.write(f"  RMSE = {row['cv_rmse_mean']:.4f} +/- {row['cv_rmse_std']:.4f}\n")
            f.write(f"  MAE = {row['cv_mae_mean']:.4f} +/- {row['cv_mae_std']:.4f}\n")

        f.write("\n\n平均性能:\n")
        f.write(f"  平均R^2 = {results_df['cv_r2_mean'].mean():.4f}\n")
        f.write(f"  平均RMSE = {results_df['cv_rmse_mean'].mean():.4f}\n")
        f.write(f"  平均MAE = {results_df['cv_mae_mean'].mean():.4f}\n")

    print(f"[OK] 报告已保存: {report_path}")

    print(f"\n[DONE] 完成！所有结果已保存到: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
