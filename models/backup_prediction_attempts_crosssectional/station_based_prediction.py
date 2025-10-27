import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = 'Province_Predictions_Fixed'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Sample_Size': len(y_true)
    }


def build_station_model(station_data, indicator, train_ratio=0.8):
    """
    Build prediction model for a specific monitoring station
    为单个监测站建立预测模型
    """
    all_indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    # Use other indicators as features
    feature_indicators = [ind for ind in all_indicators if ind != indicator]

    # Sort by date - CRITICAL!
    data = station_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)

    # Add temporal features
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['Quarter'] = data['Date'].dt.quarter

    # Add lag features (now valid because data is sorted by time)
    for lag in [1, 3, 7]:
        data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)

    # Add rolling statistics
    for window in [3, 7, 14]:
        data[f'{indicator}_rolling_mean_{window}'] = data[indicator].rolling(window=window, min_periods=1).mean()

    # Prepare feature columns
    temporal_features = ['Month', 'DayOfYear', 'Quarter']
    lag_features = [col for col in data.columns if '_lag_' in col or '_rolling_' in col]
    available_features = [feat for feat in feature_indicators if feat in data.columns]

    feature_cols = temporal_features + available_features + lag_features

    # Prepare data
    model_data = data[feature_cols + [indicator]].dropna()

    if len(model_data) < 20:
        return None

    # Split into train and test sets (80/20) - now properly temporal
    split_idx = int(len(model_data) * train_ratio)
    train_data = model_data.iloc[:split_idx]
    test_data = model_data.iloc[split_idx:]

    if len(train_data) < 15 or len(test_data) < 3:
        return None

    X_train = train_data[feature_cols].values
    y_train = train_data[indicator].values
    X_test = test_data[feature_cols].values
    y_test = test_data[indicator].values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        # Use RandomForest
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        metrics = calculate_accuracy_metrics(y_test, y_pred)

        if metrics is None:
            return None

        return {
            'model': model,
            'scaler': scaler,
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics
        }

    except Exception as e:
        return None


def predict_province_indicators(df, province):
    """
    Build models for all stations in a province
    为省份内的所有监测站建立模型
    """
    print(f"\n{'='*80}")
    print(f"Processing Province: {province}")
    print(f"{'='*80}")

    province_data = df[df['Province'] == province].copy()
    stations = province_data['Monitoring_Station'].unique()
    print(f"Total samples: {len(province_data)} | Stations: {len(stations)}")

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
        print(f"\n[{indicator}]")

        # Aggregate results from all stations
        all_y_test = []
        all_y_pred = []
        success_count = 0
        total_train = 0
        total_test = 0

        for station in stations:
            station_data = province_data[province_data['Monitoring_Station'] == station]

            if len(station_data) < 20 or station_data[indicator].notna().sum() < 15:
                continue

            result = build_station_model(station_data, indicator, train_ratio=0.8)

            if result:
                all_y_test.extend(result['y_test'])
                all_y_pred.extend(result['y_pred'])
                success_count += 1
                total_train += result['train_size']
                total_test += result['test_size']

        if len(all_y_test) > 0:
            # Calculate aggregate metrics
            metrics = calculate_accuracy_metrics(np.array(all_y_test), np.array(all_y_pred))

            if metrics:
                province_results[indicator] = {
                    'y_test': np.array(all_y_test),
                    'y_pred': np.array(all_y_pred),
                    'train_size': total_train,
                    'test_size': total_test,
                    'num_stations': success_count,
                    'metrics': metrics
                }

                print(f"  ✓ Models trained for {success_count}/{len(stations)} stations")
                print(f"    Total train: {total_train} | Total test: {total_test}")
                print(f"    R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
            else:
                print(f"  ✗ Failed to calculate metrics")
        else:
            print(f"  ✗ No valid station models")

    return province_results


def visualize_predictions(province, results, output_dir):
    """Visualize predictions vs actual values"""
    if not results:
        return

    n_indicators = len(results)
    n_cols = 4
    n_rows = (n_indicators + n_cols - 1) // n_cols

    plt.figure(figsize=(20, 5*n_rows))

    for idx, (indicator, result) in enumerate(results.items()):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        y_test = result['y_test']
        y_pred = result['y_pred']
        metrics = result['metrics']

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='#3498db', edgecolors='none')

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Metrics text
        textstr = (f"R² = {metrics['R2']:.3f}\n"
                  f"RMSE = {metrics['RMSE']:.3f}\n"
                  f"Samples = {metrics['Sample_Size']}\n"
                  f"Stations = {result['num_stations']}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{province} - Station-Based Predictions (80/20 Split)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{province}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filename}")


def generate_report(all_province_results, output_dir):
    """Generate comprehensive report"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STATION-BASED WATER QUALITY PREDICTION REPORT\n")
        f.write("基于监测站的水质预测报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: Time-series split per monitoring station (80/20)\n")
        f.write(f"Provinces: {len(all_province_results)}\n\n")

        # Overall stats
        all_r2 = []
        all_rmse = []
        all_mae = []

        for province, results in all_province_results.items():
            for indicator, result in results.items():
                all_r2.append(result['metrics']['R2'])
                all_rmse.append(result['metrics']['RMSE'])
                all_mae.append(result['metrics']['MAE'])

        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R²: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n")
        f.write(f"Median R²: {np.median(all_r2):.4f}\n")
        f.write(f"Mean RMSE: {np.mean(all_rmse):.4f}\n")
        f.write(f"Mean MAE: {np.mean(all_mae):.4f}\n")
        f.write(f"Total Predictions: {sum(r['metrics']['Sample_Size'] for results in all_province_results.values() for r in results.values())}\n\n")

        f.write("="*80 + "\n\n")

        # Province details
        for province, results in sorted(all_province_results.items()):
            if not results:
                continue

            f.write(f"PROVINCE: {province}\n")
            f.write("-"*80 + "\n")

            province_r2 = np.mean([r['metrics']['R2'] for r in results.values()])
            f.write(f"Average R²: {province_r2:.4f}\n")
            f.write(f"Indicators: {len(results)}\n\n")

            f.write(f"{'Indicator':<30} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Stations':>10} {'Samples':>10}\n")
            f.write("-"*80 + "\n")

            for indicator, result in sorted(results.items()):
                m = result['metrics']
                f.write(f"{indicator:<30} {m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} "
                       f"{result['num_stations']:>10d} {m['Sample_Size']:>10d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("INTERPRETATION:\n")
        f.write("  R² > 0.70: Good predictive performance\n")
        f.write("  R² 0.50-0.70: Moderate performance\n")
        f.write("  R² < 0.50: Poor performance\n")
        f.write("="*80 + "\n")

    print(f"\n✓ Generated: {report_path}")


def main():
    """Main execution"""
    print("="*80)
    print("STATION-BASED WATER QUALITY PREDICTION SYSTEM")
    print("基于监测站的水质预测系统")
    print("="*80)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ Loaded {len(df)} records")

    provinces = sorted(df['Province'].unique())
    print(f"✓ Provinces: {len(provinces)}")

    # Build models
    print("\n[2/4] Building station-based models...")
    all_province_results = {}

    for province in provinces:
        results = predict_province_indicators(df, province)
        if results:
            all_province_results[province] = results

    # Visualize
    print(f"\n[3/4] Generating visualizations...")
    for province, results in all_province_results.items():
        visualize_predictions(province, results, OUTPUT_DIR)

    # Report
    print(f"\n[4/4] Generating report...")
    generate_report(all_province_results, OUTPUT_DIR)

    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    total_indicators = sum(len(results) for results in all_province_results.values())
    print(f"\n✓ Trained {total_indicators} indicator models across {len(all_province_results)} provinces")
    print(f"✓ Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
