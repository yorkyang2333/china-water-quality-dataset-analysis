import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Final_Cleaned_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_data(df):
    """
    Clean the dataset by removing outliers and invalid values
    清洗数据：移除异常值和无效值
    """
    print("\n=== 数据清洗 ===")
    print(f"原始数据: {len(df)} 行")

    df_clean = df.copy()

    # 定义合理范围
    valid_ranges = {
        'Water_Temperature_C': (0, 40),        # 水温 0-40°C
        'pH': (4, 10),                         # pH 4-10
        'Dissolved_Oxygen_mg_L': (0, 20),      # 溶解氧 0-20 mg/L
        'Conductivity_uS_cm': (50, 2000),      # 电导率 50-2000 μS/cm
        'Turbidity_NTU': (0, 100),             # 浊度 0-100 NTU
        'Nitrate_mg_L': (0, 10),               # 硝酸盐 0-10 mg/L
        'Nitrite_mg_L': (0, 0.5),              # 亚硝酸盐 0-0.5 mg/L
        'Ammonia_N_mg_L': (0, 5),              # 氨氮 0-5 mg/L
        'Total_Phosphorus_mg_L': (0, 1),       # 总磷 0-1 mg/L
        'Total_Nitrogen_mg_L': (0, 10),        # 总氮 0-10 mg/L
        'COD_mg_L': (0, 100),                  # COD 0-100 mg/L
        'BOD_mg_L': (0, 50),                   # BOD 0-50 mg/L
        'Heavy_Metals_Pb_ug_L': (0, 50),       # 铅 0-50 μg/L
        'Heavy_Metals_Cd_ug_L': (0, 10),       # 镉 0-10 μg/L
        'Heavy_Metals_Hg_ug_L': (0, 2),        # 汞 0-2 μg/L
        'Coliform_Count_CFU_100mL': (0, 500)   # 大肠菌群 0-500 CFU/100mL
    }

    # 过滤异常值
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= min_val) & (df_clean[col] <= max_val)]
            removed = before - len(df_clean)
            if removed > 0:
                print(f"  {col}: 移除 {removed} 个异常值")

    print(f"清洗后数据: {len(df_clean)} 行 (移除 {len(df) - len(df_clean)} 行, {(len(df) - len(df_clean))/len(df)*100:.1f}%)")

    return df_clean


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 5:
        return None

    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Sample_Size': len(y_true)
    }


def build_model(province_data, indicator, train_ratio=0.8):
    """
    Build prediction model using all other indicators
    使用所有其他指标建立预测模型
    """
    all_indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    feature_indicators = [ind for ind in all_indicators if ind != indicator]

    # 时间特征
    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear

    # 特征列
    temporal_features = ['Month', 'DayOfYear']
    feature_cols = temporal_features + feature_indicators

    # 准备数据
    model_data = data[feature_cols + [indicator]].dropna()

    if len(model_data) < 100:
        return None

    # 时间顺序分割
    model_data = model_data.sort_index()
    split_idx = int(len(model_data) * train_ratio)
    train_data = model_data.iloc[:split_idx]
    test_data = model_data.iloc[split_idx:]

    if len(train_data) < 50 or len(test_data) < 10:
        return None

    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values
    X_test = test_data[feature_cols].values
    y_test = test_data[indicator].values

    # 使用RobustScaler处理异常值
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        # RandomForest with stronger regularization
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = calculate_metrics(y_test, y_pred)
        if metrics is None:
            return None

        # 特征重要性
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = sorted(
                zip(feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )[:5]

        return {
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics,
            'feature_importance': feature_importance
        }

    except Exception as e:
        return None


def predict_province(df, province):
    """Predict all indicators for a province"""
    print(f"\n{'='*80}")
    print(f"Province: {province}")
    print(f"{'='*80}")

    province_data = df[df['Province'] == province].copy()
    print(f"Samples: {len(province_data)}")

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    results = {}

    for indicator in indicators:
        print(f"  {indicator}...", end=' ')

        result = build_model(province_data, indicator)

        if result:
            results[indicator] = result
            m = result['metrics']
            print(f"R²={m['R2']:.3f}, RMSE={m['RMSE']:.2f}")
        else:
            print("failed")

    return results


def visualize(province, results, output_dir):
    """Visualize predictions"""
    if not results:
        return

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(20, 5*nrows))

    for idx, (indicator, result) in enumerate(results.items()):
        ax = plt.subplot(nrows, ncols, idx + 1)

        y_test = result['y_test']
        y_pred = result['y_pred']
        m = result['metrics']

        ax.scatter(y_test, y_pred, alpha=0.4, s=20, color='#3498db')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        textstr = f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{province} - Cleaned Data Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{province}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(all_results, output_dir):
    """Generate report"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WATER QUALITY PREDICTION - CLEANED DATA\n")
        f.write("水质预测报告 - 清洗后数据\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: Cross-sectional prediction with cleaned data\n")
        f.write(f"Provinces: {len(all_results)}\n\n")

        # Overall
        all_r2 = []
        all_rmse = []
        all_mae = []

        for province, results in all_results.items():
            for indicator, result in results.items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])

        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R²: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n")
        f.write(f"Median R²: {np.median(all_r2):.4f}\n")
        f.write(f"Best R²: {np.max(all_r2):.4f}\n")
        f.write(f"Worst R²: {np.min(all_r2):.4f}\n")
        f.write(f"Mean RMSE: {np.mean(all_rmse):.4f}\n")
        f.write(f"Mean MAE: {np.mean(all_mae):.4f}\n")
        f.write(f"Total Models: {len(all_r2)}\n\n")

        f.write("="*80 + "\n\n")

        # Province details
        for province, results in sorted(all_results.items()):
            if not results:
                continue

            f.write(f"PROVINCE: {province}\n")
            f.write("-"*80 + "\n")

            avg_r2 = np.mean([r['metrics']['R2'] for r in results.values()])
            f.write(f"Average R²: {avg_r2:.4f}\n")
            f.write(f"Indicators: {len(results)}\n\n")

            f.write(f"{'Indicator':<30} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Samples':>10}\n")
            f.write("-"*80 + "\n")

            for indicator, result in sorted(results.items(), key=lambda x: x[1]['metrics']['R2'], reverse=True):
                m = result['metrics']
                f.write(f"{indicator:<30} {m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>10d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("NOTES:\n")
        f.write("- Data cleaned: removed outliers outside physical/chemical valid ranges\n")
        f.write("- Used RobustScaler to handle remaining outliers\n")
        f.write("- RandomForest with strong regularization to prevent overfitting\n")
        f.write("- Results sorted by R² score (best first)\n")
        f.write("="*80 + "\n")

    print(f"\n✓ Report: {report_path}")


def main():
    """Main execution"""
    print("="*80)
    print("WATER QUALITY PREDICTION WITH DATA CLEANING")
    print("水质预测系统 - 含数据清洗")
    print("="*80)

    # Load and clean data
    print("\n[1/4] Loading and cleaning data...")
    try:
        df = pd.read_csv('中国水污染监测数据集.csv', encoding='utf-8')
    except:
        df = pd.read_csv('中国水污染监测数据集.csv', encoding='gbk')

    df_clean = clean_data(df)

    provinces = sorted(df_clean['Province'].unique())
    print(f"\n清洗后省份: {len(provinces)}")

    # Build models
    print("\n[2/4] Building models...")
    all_results = {}

    for province in provinces:
        results = predict_province(df_clean, province)
        if results:
            all_results[province] = results

    # Visualize
    print(f"\n[3/4] Generating visualizations...")
    for province, results in all_results.items():
        visualize(province, results, OUTPUT_DIR)
        print(f"  ✓ {province}")

    # Report
    print(f"\n[4/4] Generating report...")
    generate_report(all_results, OUTPUT_DIR)

    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    total = sum(len(r) for r in all_results.values())
    print(f"\n✓ Total models: {total}")
    print(f"✓ Results: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
