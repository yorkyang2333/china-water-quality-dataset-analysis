import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import warnings
import os

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = 'GTWR_Province_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_accuracy_metrics(y_true, y_pred):
    """
    Calculate comprehensive accuracy metrics
    计算综合准确度指标
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))

    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)

    # Normalized RMSE
    value_range = np.max(y_true) - np.min(y_true)
    nrmse = (rmse / (value_range + 1e-10)) * 100 if value_range > 0 else 0

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Max_Error': max_error,
        'MBE': mbe,
        'NRMSE': nrmse,
        'Sample_Size': len(y_true)
    }


def get_city_coordinates():
    """
    Get correct coordinates for each city
    获取每个城市的正确坐标
    """
    city_coords = {
        'Beijing': (116.4074, 39.9042),
        'Shanghai': (121.4737, 31.2304),
        'Hangzhou': (120.1551, 30.2741),
        'Ningbo': (121.5440, 29.8683),
        'Nanjing': (118.7969, 32.0603),
        'Suzhou': (120.5954, 31.2989),
        'Guangzhou': (113.2644, 23.1291),
        'Shenzhen': (114.0579, 22.5431),
        'Chengdu': (104.0668, 30.5728),
        'Mianyang': (104.6791, 31.4679),
        'Wuhan': (114.3055, 30.5931),
        'Yichang': (111.2868, 30.6918),
        'Jinan': (117.1205, 36.6519),
        'Qingdao': (120.3826, 36.0671),
        'Zhengzhou': (113.6254, 34.7466),
        'Luoyang': (112.4540, 34.6197),
        'Kunming': (102.8329, 24.8801),
        'Dali': (100.2251, 25.6064)
    }
    return city_coords


def build_gtwr_model(province_data, indicator, train_ratio=0.8):
    """
    Build GTWR model for a specific indicator with 80/20 train-test split
    为特定指标建立GTWR模型，按8:2划分训练集和测试集

    Parameters:
    -----------
    province_data : DataFrame
        Province-specific data
    indicator : str
        Water quality indicator to predict
    train_ratio : float
        Ratio for training data (default 0.8)

    Returns:
    --------
    dict : Model results and predictions
    """
    # All water quality indicators
    all_indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    # Use other indicators as features (exclude the target indicator)
    feature_indicators = [ind for ind in all_indicators if ind != indicator]

    # Select most correlated features (top 5)
    correlations = {}
    for feat in feature_indicators:
        if feat in province_data.columns and indicator in province_data.columns:
            valid_data = province_data[[feat, indicator]].dropna()
            if len(valid_data) > 10:
                corr = valid_data[feat].corr(valid_data[indicator])
                if not np.isnan(corr):
                    correlations[feat] = abs(corr)

    # Select top 5 most correlated features (DO NOT include Longitude/Latitude)
    if correlations:
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_cols = [f[0] for f in top_features]
    else:
        # If no correlations found, use all other indicators
        feature_cols = feature_indicators[:5]

    # Prepare data - must have City column for coordinate mapping
    required_cols = feature_cols + [indicator, 'Date', 'City']
    data = province_data[required_cols].dropna()

    if len(data) < 30:  # Minimum samples required
        return None

    # Sort by date to ensure temporal order
    data = data.sort_values('Date').reset_index(drop=True)

    # Get city coordinates
    city_coords = get_city_coordinates()

    # Map correct coordinates based on city
    data['Longitude'] = data['City'].map(lambda x: city_coords.get(x, (0, 0))[0])
    data['Latitude'] = data['City'].map(lambda x: city_coords.get(x, (0, 0))[1])

    # Remove any rows where city wasn't found
    data = data[(data['Longitude'] != 0) & (data['Latitude'] != 0)]

    if len(data) < 30:
        return None

    # Split into train and test sets (80/20)
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    if len(train_data) < 20 or len(test_data) < 5:
        return None

    # Create GeoDataFrame
    train_geom = [Point(xy) for xy in zip(train_data['Longitude'], train_data['Latitude'])]
    test_geom = [Point(xy) for xy in zip(test_data['Longitude'], test_data['Latitude'])]

    train_gdf = gpd.GeoDataFrame(train_data, geometry=train_geom)
    test_gdf = gpd.GeoDataFrame(test_data, geometry=test_geom)

    # Convert dates to numeric (days since first date)
    min_date = pd.to_datetime(train_data['Date']).min()
    train_gdf['time_numeric'] = (pd.to_datetime(train_gdf['Date']) - min_date).dt.days
    test_gdf['time_numeric'] = (pd.to_datetime(test_gdf['Date']) - min_date).dt.days

    # Prepare arrays for GTWR
    # Use feature indicators for prediction (NOT longitude/latitude)
    X_train = train_gdf[feature_cols].values
    y_train = train_gdf[indicator].values
    times_train = train_gdf['time_numeric'].values
    coords_lon_lat_train = train_gdf[['Longitude', 'Latitude']].values

    X_test = test_gdf[feature_cols].values
    y_test = test_gdf[indicator].values
    times_test = test_gdf['time_numeric'].values
    coords_lon_lat_test = test_gdf[['Longitude', 'Latitude']].values

    # Create spatio-temporal coordinates (Longitude, Latitude, Time) for GWR weighting
    coords_train = np.column_stack([coords_lon_lat_train[:, 0], coords_lon_lat_train[:, 1], times_train])
    coords_test = np.column_stack([coords_lon_lat_test[:, 0], coords_lon_lat_test[:, 1], times_test])

    # Prepare design matrix (add intercept + feature indicators)
    X_train_design = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_design = np.column_stack([np.ones(len(X_test)), X_test])

    y_train_design = y_train.reshape(-1, 1)

    try:
        # Select bandwidth using AIC
        print(f"    Selecting optimal bandwidth...")
        bw_selector = Sel_BW(coords_train, y_train_design, X_train_design,
                            kernel='gaussian', fixed=False)
        bw = bw_selector.search()

        print(f"    Optimal bandwidth: {bw:.4f}")

        # Fit GTWR model (GWR with 3D spatio-temporal coordinates)
        print(f"    Fitting GTWR model...")
        gtwr_model = GWR(coords_train, y_train_design, X_train_design,
                        bw=bw, kernel='gaussian', fixed=False)
        gtwr_results = gtwr_model.fit()

        # Predict on test set
        print(f"    Making predictions on test set...")
        # Calculate predictions using the training model parameters
        # We need to manually compute predictions based on local weights
        y_pred = []
        for i in range(len(coords_test)):
            # Calculate distances from test point to all training points
            test_coord = coords_test[i:i+1]

            # Calculate weights using Gaussian kernel
            distances = np.sqrt(np.sum((coords_train - test_coord)**2, axis=1))
            weights = np.exp(-0.5 * (distances / bw)**2)

            # Weighted least squares for this point
            W = np.diag(weights)
            try:
                XtWX_inv = np.linalg.inv(X_train_design.T @ W @ X_train_design)
                beta_local = XtWX_inv @ X_train_design.T @ W @ y_train.reshape(-1, 1)
                y_pred_i = X_test_design[i:i+1] @ beta_local
                y_pred.append(y_pred_i[0, 0])
            except:
                # If singular matrix, use mean prediction
                y_pred.append(np.mean(y_train))

        y_pred = np.array(y_pred)

        # Calculate accuracy metrics
        metrics = calculate_accuracy_metrics(y_test, y_pred)

        if metrics is None:
            return None

        # Store results
        results = {
            'model': gtwr_model,
            'results': gtwr_results,
            'bandwidth': bw,
            'features': feature_cols,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'train_dates': train_gdf['Date'].values,
            'test_dates': test_gdf['Date'].values,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics
        }

        return results

    except Exception as e:
        print(f"    Error: {str(e)}")
        return None


def predict_province_indicators(df, province):
    """
    Build GTWR models for all water quality indicators in a province
    为省份的所有水质指标建立GTWR模型

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    province : str
        Province name

    Returns:
    --------
    dict : Results for all indicators
    """
    print(f"\n{'='*80}")
    print(f"Processing Province: {province}")
    print(f"{'='*80}")

    province_data = df[df['Province'] == province].copy()
    print(f"Total samples: {len(province_data)}")

    # Water quality indicators (continuous variables)
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

        # Check if indicator has enough valid data
        valid_count = province_data[indicator].notna().sum()
        if valid_count < 30:
            print(f"  ✗ Insufficient data (only {valid_count} valid samples)")
            continue

        # Build GTWR model
        results = build_gtwr_model(province_data, indicator, train_ratio=0.8)

        if results:
            province_results[indicator] = results
            metrics = results['metrics']
            print(f"  ✓ Model trained successfully")
            print(f"    Features used: {', '.join([f.replace('_', ' ') for f in results['features']])}")
            print(f"    Train samples: {results['train_size']} | Test samples: {results['test_size']}")
            print(f"    R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
        else:
            print(f"  ✗ Model training failed")

    return province_results


def visualize_predictions(province, results, output_dir):
    """
    Visualize predictions vs actual values for all indicators
    可视化所有指标的预测值与实际值对比
    """
    if not results:
        return

    n_indicators = len(results)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols

    plt.figure(figsize=(18, 5*n_rows))

    for idx, (indicator, result) in enumerate(results.items()):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        y_test = result['y_test']
        y_pred = result['y_pred']
        metrics = result['metrics']

        # Scatter plot: Predicted vs Actual
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='#3498db', edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Add metrics text
        textstr = (f"R² = {metrics['R2']:.3f}\n"
                  f"RMSE = {metrics['RMSE']:.3f}\n"
                  f"MAE = {metrics['MAE']:.3f}\n"
                  f"Samples = {metrics['Sample_Size']}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_xlabel('Actual Values', fontsize=10)
        ax.set_ylabel('Predicted Values', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'GTWR Model Predictions - {province} (80/20 Train-Test Split)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f'{output_dir}/{province}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved prediction visualization: {filename}")


def save_predictions_to_csv(province, results, output_dir):
    """
    Save prediction results to CSV file
    保存预测结果到CSV文件
    """
    if not results:
        return

    all_predictions = []

    for indicator, result in results.items():
        test_dates = result['test_dates']
        y_test = result['y_test']
        y_pred = result['y_pred']

        for i in range(len(y_test)):
            all_predictions.append({
                'Province': province,
                'Date': test_dates[i],
                'Indicator': indicator,
                'Actual_Value': y_test[i],
                'Predicted_Value': y_pred[i],
                'Error': y_pred[i] - y_test[i],
                'Absolute_Error': abs(y_pred[i] - y_test[i])
            })

    pred_df = pd.DataFrame(all_predictions)
    filename = f'{output_dir}/{province}_predictions.csv'
    pred_df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"✓ Saved predictions to CSV: {filename}")


def generate_comprehensive_report(all_province_results, output_dir):
    """
    Generate comprehensive evaluation report for all provinces
    生成所有省份的综合评估报告
    """
    report_path = f'{output_dir}/comprehensive_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GTWR MODEL PREDICTION REPORT\n")
        f.write("GTWR模型预测报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Split: 80% Training / 20% Testing\n")
        f.write(f"Total Provinces Analyzed: {len(all_province_results)}\n\n")

        # Overall statistics
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_mape = []

        for province, results in all_province_results.items():
            for indicator, result in results.items():
                metrics = result['metrics']
                all_r2.append(metrics['R2'])
                all_rmse.append(metrics['RMSE'])
                all_mae.append(metrics['MAE'])
                all_mape.append(metrics['MAPE'])

        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R² Score: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n")
        f.write(f"Median R² Score: {np.median(all_r2):.4f}\n")
        f.write(f"Mean RMSE: {np.mean(all_rmse):.4f} (±{np.std(all_rmse):.4f})\n")
        f.write(f"Mean MAE: {np.mean(all_mae):.4f} (±{np.std(all_mae):.4f})\n")
        f.write(f"Mean MAPE: {np.mean(all_mape):.2f}% (±{np.std(all_mape):.2f}%)\n")
        f.write(f"Total Indicators Evaluated: {len(all_r2)}\n\n")

        f.write("="*80 + "\n\n")

        # Province-specific results
        for province, results in sorted(all_province_results.items()):
            if not results:
                continue

            f.write(f"PROVINCE: {province}\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Indicators: {len(results)}\n\n")

            # Province average metrics
            province_r2 = np.mean([r['metrics']['R2'] for r in results.values()])
            province_rmse = np.mean([r['metrics']['RMSE'] for r in results.values()])
            province_mae = np.mean([r['metrics']['MAE'] for r in results.values()])
            province_mape = np.mean([r['metrics']['MAPE'] for r in results.values()])

            f.write(f"Province Average Performance:\n")
            f.write(f"  R² Score: {province_r2:.4f}\n")
            f.write(f"  RMSE: {province_rmse:.4f}\n")
            f.write(f"  MAE: {province_mae:.4f}\n")
            f.write(f"  MAPE: {province_mape:.2f}%\n\n")

            f.write(f"{'Indicator':<35} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10} {'Train':>8} {'Test':>8}\n")
            f.write("-"*80 + "\n")

            for indicator, result in sorted(results.items()):
                metrics = result['metrics']
                f.write(f"{indicator:<35} "
                       f"{metrics['R2']:>8.4f} "
                       f"{metrics['RMSE']:>10.4f} "
                       f"{metrics['MAE']:>10.4f} "
                       f"{metrics['MAPE']:>10.2f} "
                       f"{result['train_size']:>8d} "
                       f"{result['test_size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        # Performance interpretation
        f.write("PERFORMANCE INTERPRETATION GUIDE\n")
        f.write("-"*80 + "\n")
        f.write("R² Score (Coefficient of Determination):\n")
        f.write("  > 0.90: Excellent predictive performance\n")
        f.write("  0.70-0.90: Good predictive performance\n")
        f.write("  0.50-0.70: Moderate predictive performance\n")
        f.write("  < 0.50: Poor predictive performance\n\n")

        f.write("RMSE (Root Mean Square Error):\n")
        f.write("  Lower values indicate better accuracy\n")
        f.write("  Sensitive to outliers\n\n")

        f.write("MAE (Mean Absolute Error):\n")
        f.write("  Average absolute difference between predicted and actual values\n")
        f.write("  More robust to outliers than RMSE\n\n")

        f.write("MAPE (Mean Absolute Percentage Error):\n")
        f.write("  Percentage-based error metric\n")
        f.write("  < 10%: Highly accurate\n")
        f.write("  10-20%: Good accuracy\n")
        f.write("  20-50%: Reasonable accuracy\n")
        f.write("  > 50%: Poor accuracy\n\n")

        f.write("="*80 + "\n")

    print(f"\n✓ Generated comprehensive report: {report_path}")


def main():
    """
    Main execution function
    主执行函数
    """
    print("="*80)
    print("GTWR MODEL - PROVINCE-WISE WATER QUALITY PREDICTION SYSTEM")
    print("GTWR模型 - 分省水质指标预测系统")
    print("="*80)

    # Load data
    print("\n[Step 1/5] Loading preprocessed data...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    provinces = sorted(df['Province'].unique())
    print(f"✓ Found {len(provinces)} provinces: {', '.join(provinces)}")

    # Build GTWR models for each province
    print("\n[Step 2/5] Building GTWR models for each province...")
    all_province_results = {}

    for province in provinces:
        province_results = predict_province_indicators(df, province)
        if province_results:
            all_province_results[province] = province_results

    # Visualize predictions
    print(f"\n[Step 3/5] Generating prediction visualizations...")
    for province, results in all_province_results.items():
        visualize_predictions(province, results, OUTPUT_DIR)

    # Save predictions to CSV
    print(f"\n[Step 4/5] Saving predictions to CSV files...")
    for province, results in all_province_results.items():
        save_predictions_to_csv(province, results, OUTPUT_DIR)

    # Generate comprehensive report
    print(f"\n[Step 5/5] Generating comprehensive report...")
    generate_comprehensive_report(all_province_results, OUTPUT_DIR)

    # Summary
    print("\n" + "="*80)
    print("PREDICTION PROCESS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"  - Province prediction charts: {len(all_province_results)} files")
    print(f"  - Province prediction data (CSV): {len(all_province_results)} files")
    print(f"  - Comprehensive report: comprehensive_report.txt")

    total_indicators = sum(len(results) for results in all_province_results.values())
    print(f"\n✓ Successfully trained {total_indicators} GTWR models across {len(all_province_results)} provinces")
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()
