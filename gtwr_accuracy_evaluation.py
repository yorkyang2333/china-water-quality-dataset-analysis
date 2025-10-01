import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = 'GTWR_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GTWR:
    """
    Geographically and Temporally Weighted Regression (GTWR) Model
    """

    def __init__(self, spatial_bandwidth=1.0, temporal_bandwidth=30,
                 spatial_weight=0.5, temporal_weight=0.5):
        self.spatial_bandwidth = spatial_bandwidth
        self.temporal_bandwidth = temporal_bandwidth
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def spatial_distance(self, lat1, lon1, lat2, lon2):
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    def temporal_distance(self, t1, t2):
        return np.abs((t1 - t2).days)

    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-(distance**2) / (2 * bandwidth**2))

    def calculate_weights(self, pred_lat, pred_lon, pred_time,
                          train_lats, train_lons, train_times):
        n = len(train_lats)
        weights = np.zeros(n)

        for i in range(n):
            spatial_dist = self.spatial_distance(pred_lat, pred_lon,
                                                 train_lats[i], train_lons[i])
            spatial_w = self.gaussian_kernel(spatial_dist, self.spatial_bandwidth)

            temporal_dist = self.temporal_distance(pred_time, train_times[i])
            temporal_w = self.gaussian_kernel(temporal_dist, self.temporal_bandwidth)

            weights[i] = (self.spatial_weight * spatial_w +
                         self.temporal_weight * temporal_w)

        weights = weights / (np.sum(weights) + 1e-10)
        return weights

    def fit(self, X, y, latitudes, longitudes, times):
        self.X_train = np.array(X)
        self.y_train = np.array(y).reshape(-1, 1)
        self.train_lats = np.array(latitudes)
        self.train_lons = np.array(longitudes)
        self.train_times = times

        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)

    def predict_point(self, X_pred, lat, lon, time):
        weights = self.calculate_weights(lat, lon, time,
                                        self.train_lats,
                                        self.train_lons,
                                        self.train_times)

        X_pred_scaled = self.scaler_X.transform(X_pred.reshape(1, -1))

        X_train_with_intercept = np.c_[np.ones(len(self.X_train_scaled)),
                                       self.X_train_scaled]
        X_pred_with_intercept = np.c_[np.ones(1), X_pred_scaled]

        W = np.diag(weights)
        try:
            XtWX = X_train_with_intercept.T @ W @ X_train_with_intercept
            XtWy = X_train_with_intercept.T @ W @ self.y_train_scaled
            beta = np.linalg.solve(XtWX + 1e-6 * np.eye(XtWX.shape[0]), XtWy)

            y_pred_scaled = X_pred_with_intercept @ beta
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

            return y_pred[0, 0]
        except:
            return np.sum(weights * self.y_train.flatten())

    def predict(self, X_pred, lats, lons, times):
        predictions = []
        for i in range(len(X_pred)):
            pred = self.predict_point(X_pred[i], lats[i], lons[i], times[i])
            predictions.append(pred)
        return np.array(predictions)


def calculate_metrics(y_true, y_pred):
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
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))

    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)

    # Normalized RMSE
    nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-10) * 100

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


def cross_validate_indicator(df, province, indicator, n_splits=5):
    """
    Perform time series cross-validation for a specific indicator
    对特定指标进行时间序列交叉验证
    """
    province_data = df[df['Province'] == province].copy()
    feature_cols = ['Latitude', 'Longitude', 'Water_Quality_Index']

    # Prepare data
    train_data = province_data[feature_cols + [indicator, 'Date']].dropna()

    if len(train_data) < 50:
        return None

    train_data = train_data.sort_values('Date').reset_index(drop=True)

    X = train_data[feature_cols].values
    y = train_data[indicator].values
    lats = train_data['Latitude'].values
    lons = train_data['Longitude'].values
    dates = train_data['Date'].values
    times = [pd.Timestamp(d) for d in dates]

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        lats_train, lats_test = lats[train_idx], lats[test_idx]
        lons_train, lons_test = lons[train_idx], lons[test_idx]
        times_train, times_test = [times[i] for i in train_idx], [times[i] for i in test_idx]

        # Train model
        model = GTWR(spatial_bandwidth=0.5, temporal_bandwidth=30,
                    spatial_weight=0.4, temporal_weight=0.6)

        model.fit(X_train, y_train, lats_train, lons_train, times_train)

        # Predict
        y_pred = model.predict(X_test, lats_test, lons_test, times_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Calculate metrics
    metrics = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))

    if metrics:
        metrics['y_true'] = np.array(all_y_true)
        metrics['y_pred'] = np.array(all_y_pred)

    return metrics


def evaluate_province(df, province):
    """
    Evaluate GTWR model accuracy for all indicators in a province
    评估省份所有指标的GTWR模型准确度
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Province: {province}")
    print(f"{'='*80}")

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
        print(f"\nEvaluating {indicator}...")
        try:
            metrics = cross_validate_indicator(df, province, indicator, n_splits=5)
            if metrics:
                results[indicator] = metrics
                print(f"  ✓ RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
            else:
                print(f"  ✗ Insufficient data")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

    return results


def visualize_accuracy(province, results, output_dir):
    """
    Visualize prediction accuracy with scatter plots
    可视化预测准确度（散点图）
    """
    if not results:
        return

    n_indicators = len(results)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_indicators > 1 else [axes]

    for idx, (indicator, metrics) in enumerate(results.items()):
        ax = axes[idx]

        y_true = metrics['y_true']
        y_pred = metrics['y_pred']

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='#3498db')

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Add metrics text
        textstr = f"R² = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.3f}\nMAE = {metrics['MAE']:.3f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(indicator.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xlabel('True Values', fontsize=9)
        ax.set_ylabel('Predicted Values', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_indicators, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'GTWR Model Accuracy Evaluation - {province}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f'{output_dir}/{province}_accuracy.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved accuracy visualization: {filename}")


def generate_accuracy_report(all_results, output_dir):
    """
    Generate comprehensive accuracy evaluation report
    生成综合准确度评估报告
    """
    report_path = f'{output_dir}/accuracy_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GTWR MODEL ACCURACY EVALUATION REPORT\n")
        f.write("GTWR模型准确度评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation Method: 5-Fold Time Series Cross-Validation\n")
        f.write(f"Total Provinces Evaluated: {len(all_results)}\n\n")

        # Overall statistics
        all_r2 = []
        all_rmse = []
        all_mae = []

        for province, results in all_results.items():
            for indicator, metrics in results.items():
                all_r2.append(metrics['R2'])
                all_rmse.append(metrics['RMSE'])
                all_mae.append(metrics['MAE'])

        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R² Score: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n")
        f.write(f"Mean RMSE: {np.mean(all_rmse):.4f} (±{np.std(all_rmse):.4f})\n")
        f.write(f"Mean MAE: {np.mean(all_mae):.4f} (±{np.std(all_mae):.4f})\n")
        f.write(f"Total Indicators Evaluated: {len(all_r2)}\n\n")

        f.write("="*80 + "\n\n")

        # Province-specific results
        for province, results in all_results.items():
            if not results:
                continue

            f.write(f"PROVINCE: {province}\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Indicators Evaluated: {len(results)}\n\n")

            # Calculate province average
            province_r2 = np.mean([m['R2'] for m in results.values()])
            province_rmse = np.mean([m['RMSE'] for m in results.values()])
            province_mae = np.mean([m['MAE'] for m in results.values()])

            f.write(f"Province Average Performance:\n")
            f.write(f"  R² Score: {province_r2:.4f}\n")
            f.write(f"  RMSE: {province_rmse:.4f}\n")
            f.write(f"  MAE: {province_mae:.4f}\n\n")

            f.write(f"{'Indicator':<30} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10} {'Samples':>10}\n")
            f.write("-"*80 + "\n")

            for indicator, metrics in results.items():
                f.write(f"{indicator:<30} "
                       f"{metrics['R2']:>8.4f} "
                       f"{metrics['RMSE']:>10.4f} "
                       f"{metrics['MAE']:>10.4f} "
                       f"{metrics['MAPE']:>10.2f} "
                       f"{metrics['Sample_Size']:>10d}\n")

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
        f.write("  Should be interpreted in context of the indicator's scale\n\n")

        f.write("MAE (Mean Absolute Error):\n")
        f.write("  Average absolute difference between predicted and actual values\n")
        f.write("  More interpretable than RMSE, less sensitive to outliers\n\n")

        f.write("="*80 + "\n")

    print(f"\n✓ Generated accuracy report: {report_path}")


def main():
    """
    Main execution function
    """
    print("="*80)
    print("GTWR MODEL ACCURACY EVALUATION SYSTEM")
    print("GTWR模型准确度评估系统")
    print("="*80)

    # Load data
    print("\n[1/4] Loading preprocessed data...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"✓ Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

    provinces = df['Province'].unique()
    print(f"✓ Found {len(provinces)} provinces")

    # Evaluate each province
    print("\n[2/4] Evaluating model accuracy for each province...")
    all_results = {}

    for province in provinces:
        results = evaluate_province(df, province)
        if results:
            all_results[province] = results

    # Visualize results
    print("\n[3/4] Generating accuracy visualizations...")
    for province, results in all_results.items():
        visualize_accuracy(province, results, OUTPUT_DIR)

    # Generate report
    print("\n[4/4] Generating accuracy evaluation report...")
    generate_accuracy_report(all_results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("ACCURACY EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"  - Province-specific accuracy charts: {len(all_results)} files")
    print(f"  - Comprehensive accuracy report: accuracy_report.txt")
    print("\n✓ All evaluations completed successfully!")


if __name__ == "__main__":
    main()
