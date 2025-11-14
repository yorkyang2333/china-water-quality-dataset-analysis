"""
水质指标回归预测 - XGBoost方法
Water Quality Indicator Regression using XGBoost

⭐⭐⭐ 论文主要方法 (Paper Main Method)

核心思想：
  使用其他水质指标预测目标指标（不依赖时间序列）
  适用于不规则采样数据（ACF≈0时的有效替代）

预期性能：
  R² > 0.5（远超时间序列方法的R²<0）
  可用于实际预测和论文对比实验

作者：Claude Code
日期：2025-11-13
用途：论文实验 + 实际应用
"""

import zhplot
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
OUTPUT_DIR = 'models_outputs/indicator_regression'
PAPER_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

# 目标指标列表（用于预测）
TARGET_INDICATORS = [
    'COD_mg_L',                    # 化学需氧量
    'BOD_mg_L',                    # 生化需氧量
    'Dissolved_Oxygen_mg_L',       # 溶解氧
    'Total_Nitrogen_mg_L',         # 总氮
    'Total_Phosphorus_mg_L',       # 总磷
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
def engineer_features(data, target_indicator):
    """
    特征工程（横截面方法）

    特征包括：
    - 其他水质指标（物理化学关系）
    - 时间特征（月份、季度、循环编码）
    - 空间特征（省份）
    - 工程特征（比值等）

    不使用：
    - 滞后特征（因为ACF≈0）
    - 滚动统计（因为不规则采样）
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

    # 2. 工程特征
    for feature in ENGINEERED_FEATURES:
        if feature in df.columns:
            feature_cols.append(feature)

    # 3. 时间特征
    feature_cols.extend(['Month', 'Quarter', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy', 'IsWeekend'])

    # 4. 空间特征
    feature_cols.extend(['Province_encoded'])

    # 如果有 station_count 特征，也加入
    if 'station_count' in df.columns:
        feature_cols.append('station_count')

    # 移除缺失值
    df = df.dropna(subset=feature_cols + [target_indicator])

    print(f"[OK] 特征数量: {len(feature_cols)}")
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
    axes[0].set_title(f'{target_indicator} 预测结果\nR² = {results["final_r2"]:.4f}, RMSE = {results["final_rmse"]:.4f}',
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
    save_path = f"{OUTPUT_DIR}/{target_indicator}_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 预测图已保存: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_cols, target_indicator, top_n=15):
    """绘制特征重要性"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['Importance'].values, alpha=0.8)
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'].values)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title(f'{target_indicator} - 特征重要性分析 (Top {top_n})',
             fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    save_path = f"{OUTPUT_DIR}/{target_indicator}_feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 特征重要性图已保存: {save_path}")
    plt.close()

    return feature_importance

# ==================== 主程序 ====================
def main():
    print("\n" + "="*70)
    print("  水质指标回归预测 - XGBoost方法")
    print("  论文主要方法 (Paper Main Method)")
    print("="*70)

    # 加载数据
    data = load_data()
    if data is None:
        return

    # 存储所有结果
    all_results = []

    # 对每个目标指标进行预测
    for target_indicator in TARGET_INDICATORS:
        print(f"\n{'='*70}")
        print(f"[TARGET] 目标指标: {target_indicator}")
        print(f"{'='*70}")

        try:
            # 特征工程
            df, feature_cols, le_province = engineer_features(data, target_indicator)

            # 训练和评估
            model, results, y_pred = train_and_evaluate(df, target_indicator, feature_cols, cv_folds=5)

            # 可视化
            y_true = df[target_indicator].values
            plot_predictions(y_true, y_pred, target_indicator, results)
            feature_importance = plot_feature_importance(model, feature_cols, target_indicator, top_n=15)

            # 保存特征重要性到CSV
            importance_path = f"{OUTPUT_DIR}/{target_indicator}_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
            print(f"[OK] 特征重要性已保存: {importance_path}")

            all_results.append(results)

        except Exception as e:
            print(f"[ERROR] 处理 {target_indicator} 时出错: {e}")
            continue

    # 生成总结报告
    print(f"\n{'='*70}")
    print("  总结报告")
    print(f"{'='*70}")

    results_df = pd.DataFrame(all_results)

    # 控制台输出
    print("\n交叉验证结果汇总:")
    print(results_df[['target', 'cv_r2_mean', 'cv_r2_std', 'cv_rmse_mean', 'cv_mae_mean']].to_string(index=False))

    # 保存到CSV
    results_path = f"{OUTPUT_DIR}/regression_results_summary.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 结果已保存: {results_path}")

    # 生成报告文件
    report_path = f"{OUTPUT_DIR}/regression_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("水质指标回归预测 - XGBoost方法\n")
        f.write("="*70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: XGBoost回归 + 5折交叉验证\n")
        f.write(f"目标指标数量: {len(TARGET_INDICATORS)}\n")
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

    # 生成对比图（论文用）
    plot_comparison_chart(results_df)

    print(f"\n[DONE] 完成！所有结果已保存到: {OUTPUT_DIR}/")

def plot_comparison_chart(results_df):
    """生成方法对比柱状图（论文用）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R²对比
    axes[0].bar(range(len(results_df)), results_df['cv_r2_mean'],
               yerr=results_df['cv_r2_std'], alpha=0.7, capsize=5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=1, label='基线 (R²=0)')
    axes[0].axhline(y=0.5, color='g', linestyle='--', lw=1, label='目标 (R²=0.5)')
    axes[0].set_xticks(range(len(results_df)))
    axes[0].set_xticklabels([t.replace('_mg_L', '').replace('_', ' ')
                              for t in results_df['target']],
                             rotation=45, ha='right')
    axes[0].set_ylabel('R² 分数', fontsize=12)
    axes[0].set_title('XGBoost回归 - R²性能对比', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # RMSE对比
    axes[1].bar(range(len(results_df)), results_df['cv_rmse_mean'],
               yerr=results_df['cv_rmse_std'], alpha=0.7, capsize=5, color='orange')
    axes[1].set_xticks(range(len(results_df)))
    axes[1].set_xticklabels([t.replace('_mg_L', '').replace('_', ' ')
                              for t in results_df['target']],
                             rotation=45, ha='right')
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('XGBoost回归 - RMSE性能对比', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = f"{PAPER_DIR}/xgboost_regression_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 对比图已保存（论文用）: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
