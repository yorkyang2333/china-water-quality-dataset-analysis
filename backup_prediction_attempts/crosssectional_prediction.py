import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Province_CrossSectional_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def build_province_model(province_data, indicator, train_ratio=0.8):
    """
    Build cross-sectional prediction model
    横截面预测模型（使用所有其他指标作为特征）
    """
    all_indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    # 使用其他所有指标作为特征
    feature_indicators = [ind for ind in all_indicators if ind != indicator]

    # 添加时间特征
    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Season'] = data['Date'].dt.month % 12 // 3 + 1  # 1=冬, 2=春, 3=夏, 4=秋
    data['DayOfYear'] = data['Date'].dt.dayofyear

    # 特征列
    temporal_features = ['Month', 'Season', 'DayOfYear']
    feature_cols = temporal_features + feature_indicators

    # 准备数据
    model_data = data[feature_cols + [indicator]].dropna()

    if len(model_data) < 100:
        return None

    # 按时间排序后分割（前80%训练，后20%测试）
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

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        # 尝试多个模型
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }

        best_r2 = -np.inf
        best_model = None
        best_model_name = None
        best_pred = None

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = model_name
                best_pred = y_pred

        # 计算特征重要性
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feature_importance = sorted(
                zip(feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )[:10]

        metrics = calculate_metrics(y_test, best_pred)
        if metrics is None:
            return None

        return {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': scaler,
            'features': feature_cols,
            'feature_importance': feature_importance,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': best_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics
        }

    except Exception as e:
        print(f"      Error: {str(e)}")
        return None


def predict_province(df, province):
    """Predict all indicators for a province"""
    print(f"\n{'='*80}")
    print(f"Province: {province}")
    print(f"{'='*80}")

    province_data = df[df['Province'] == province].copy()
    print(f"Total samples: {len(province_data)}")

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    province_results = {}

    for indicator in indicators:
        print(f"\n  [{indicator}]")

        valid_count = province_data[indicator].notna().sum()
        if valid_count < 100:
            print(f"    ✗ Insufficient data ({valid_count} samples)")
            continue

        result = build_province_model(province_data, indicator)

        if result:
            province_results[indicator] = result
            m = result['metrics']
            print(f"    ✓ {result['model_name']}")
            print(f"      Train: {result['train_size']} | Test: {result['test_size']}")
            print(f"      R²: {m['R2']:.4f} | RMSE: {m['RMSE']:.4f} | MAE: {m['MAE']:.4f}")

            if result['feature_importance']:
                top3 = ', '.join([f[0].replace('_', ' ')[:15] for f in result['feature_importance'][:3]])
                print(f"      Top features: {top3}")
        else:
            print(f"    ✗ Failed")

    return province_results


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

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#3498db')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        textstr = f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{province} - Cross-Sectional Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{province}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {filename}")


def generate_report(all_results, output_dir):
    """Generate report"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-SECTIONAL WATER QUALITY PREDICTION REPORT\n")
        f.write("横截面水质预测报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: Cross-sectional with temporal features (80/20 split)\n")
        f.write(f"Provinces: {len(all_results)}\n\n")

        # Overall
        all_r2 = []
        all_rmse = []
        all_mae = []
        model_counts = {}

        for province, results in all_results.items():
            for indicator, result in results.items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                model_counts[result['model_name']] = model_counts.get(result['model_name'], 0) + 1

        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R²: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n")
        f.write(f"Median R²: {np.median(all_r2):.4f}\n")
        f.write(f"Best R²: {np.max(all_r2):.4f}\n")
        f.write(f"Mean RMSE: {np.mean(all_rmse):.4f}\n")
        f.write(f"Mean MAE: {np.mean(all_mae):.4f}\n")
        f.write(f"Total Models: {len(all_r2)}\n\n")

        f.write("Model Usage:\n")
        for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model_name}: {count} ({count/len(all_r2)*100:.1f}%)\n")
        f.write("\n" + "="*80 + "\n\n")

        # Province details
        for province, results in sorted(all_results.items()):
            if not results:
                continue

            f.write(f"PROVINCE: {province}\n")
            f.write("-"*80 + "\n")

            avg_r2 = np.mean([r['metrics']['R2'] for r in results.values()])
            f.write(f"Average R²: {avg_r2:.4f}\n")
            f.write(f"Indicators: {len(results)}\n\n")

            f.write(f"{'Indicator':<30} {'Model':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Samples':>10}\n")
            f.write("-"*80 + "\n")

            for indicator, result in sorted(results.items()):
                m = result['metrics']
                f.write(f"{indicator:<30} {result['model_name']:<20} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>10d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("NOTES:\n")
        f.write("- This uses cross-sectional prediction (other water quality indicators\n")
        f.write("  as features) rather than time-series prediction\n")
        f.write("- Each indicator is predicted using all other indicators + temporal features\n")
        f.write("- Train/test split is chronological (first 80% vs last 20%)\n")
        f.write("="*80 + "\n")

    print(f"\n  ✓ Report: {report_path}")


def main():
    """Main execution"""
    print("="*80)
    print("CROSS-SECTIONAL WATER QUALITY PREDICTION")
    print("横截面水质预测系统")
    print("="*80)

    print("\n[1/4] Loading data...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"  ✓ Loaded {len(df)} records")

    provinces = sorted(df['Province'].unique())
    print(f"  ✓ Provinces: {len(provinces)}")

    print("\n[2/4] Building models...")
    all_results = {}

    for province in provinces:
        results = predict_province(df, province)
        if results:
            all_results[province] = results

    print(f"\n[3/4] Generating visualizations...")
    for province, results in all_results.items():
        visualize(province, results, OUTPUT_DIR)

    print(f"\n[4/4] Generating report...")
    generate_report(all_results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    total = sum(len(r) for r in all_results.values())
    print(f"\n✓ Total models: {total}")
    print(f"✓ Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
