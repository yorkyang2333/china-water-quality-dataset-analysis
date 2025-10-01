import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
OUTPUT_DIR = 'GTWR_Predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GTWR:
    """
    Geographically and Temporally Weighted Regression (GTWR) Model
    时空地理加权回归模型
    """

    def __init__(self, spatial_bandwidth=1.0, temporal_bandwidth=30,
                 spatial_weight=0.5, temporal_weight=0.5):
        """
        Initialize GTWR model

        Parameters:
        -----------
        spatial_bandwidth : float
            Spatial bandwidth for Gaussian kernel (in degrees)
        temporal_bandwidth : int
            Temporal bandwidth (in days)
        spatial_weight : float
            Weight for spatial distance (0-1)
        temporal_weight : float
            Weight for temporal distance (0-1)
        """
        self.spatial_bandwidth = spatial_bandwidth
        self.temporal_bandwidth = temporal_bandwidth
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def spatial_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate spatial distance using Euclidean distance
        计算空间距离（欧氏距离）
        """
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    def temporal_distance(self, t1, t2):
        """
        Calculate temporal distance in days
        计算时间距离（天数）
        """
        return np.abs((t1 - t2).days)

    def gaussian_kernel(self, distance, bandwidth):
        """
        Gaussian kernel function
        高斯核函数
        """
        return np.exp(-(distance**2) / (2 * bandwidth**2))

    def calculate_weights(self, pred_lat, pred_lon, pred_time,
                          train_lats, train_lons, train_times):
        """
        Calculate spatio-temporal weights
        计算时空权重
        """
        n = len(train_lats)
        weights = np.zeros(n)

        for i in range(n):
            # Spatial weight
            spatial_dist = self.spatial_distance(pred_lat, pred_lon,
                                                 train_lats[i], train_lons[i])
            spatial_w = self.gaussian_kernel(spatial_dist, self.spatial_bandwidth)

            # Temporal weight
            temporal_dist = self.temporal_distance(pred_time, train_times[i])
            temporal_w = self.gaussian_kernel(temporal_dist, self.temporal_bandwidth)

            # Combined weight
            weights[i] = (self.spatial_weight * spatial_w +
                         self.temporal_weight * temporal_w)

        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-10)

        return weights

    def fit(self, X, y, latitudes, longitudes, times):
        """
        Fit the GTWR model
        训练模型
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).reshape(-1, 1)
        self.train_lats = np.array(latitudes)
        self.train_lons = np.array(longitudes)
        self.train_times = times

        # Standardize features
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)

    def predict_point(self, X_pred, lat, lon, time):
        """
        Predict a single point
        预测单个点
        """
        # Calculate weights
        weights = self.calculate_weights(lat, lon, time,
                                        self.train_lats,
                                        self.train_lons,
                                        self.train_times)

        # Standardize input
        X_pred_scaled = self.scaler_X.transform(X_pred.reshape(1, -1))

        # Add intercept
        X_train_with_intercept = np.c_[np.ones(len(self.X_train_scaled)),
                                       self.X_train_scaled]
        X_pred_with_intercept = np.c_[np.ones(1), X_pred_scaled]

        # Weighted least squares
        W = np.diag(weights)
        try:
            # (X'WX)^-1 X'Wy
            XtWX = X_train_with_intercept.T @ W @ X_train_with_intercept
            XtWy = X_train_with_intercept.T @ W @ self.y_train_scaled
            beta = np.linalg.solve(XtWX + 1e-6 * np.eye(XtWX.shape[0]), XtWy)

            # Prediction
            y_pred_scaled = X_pred_with_intercept @ beta
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

            return y_pred[0, 0]
        except:
            # Fallback to weighted average
            return np.sum(weights * self.y_train.flatten())

    def predict(self, X_pred, lats, lons, times):
        """
        Predict multiple points
        预测多个点
        """
        predictions = []
        for i in range(len(X_pred)):
            pred = self.predict_point(X_pred[i], lats[i], lons[i], times[i])
            predictions.append(pred)
        return np.array(predictions)


def load_and_prepare_data(filepath):
    """
    Load and prepare data for GTWR modeling
    加载并准备数据
    """
    # Load data
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        df = pd.read_csv(filepath, encoding='gbk')

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    return df


def predict_province_indicators(df, province, forecast_days=30):
    """
    Predict water quality indicators for a specific province
    预测特定省份的水质指标
    """
    print(f"\n{'='*80}")
    print(f"Predicting for Province: {province}")
    print(f"{'='*80}")

    # Filter data for the province
    province_data = df[df['Province'] == province].copy()

    if len(province_data) < 50:
        print(f"Warning: Insufficient data for {province} (only {len(province_data)} records)")
        return None

    # Define continuous water quality indicators
    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    # Prepare features for prediction
    feature_cols = ['Latitude', 'Longitude', 'Water_Quality_Index']

    results = {}

    # Get the last date and predict future dates
    last_date = province_data['Date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

    # Use mean location for province
    mean_lat = province_data['Latitude'].mean()
    mean_lon = province_data['Longitude'].mean()

    # Train and predict for each indicator
    for indicator in indicators:
        print(f"\nProcessing indicator: {indicator}")

        try:
            # Prepare training data
            train_data = province_data[feature_cols + [indicator]].dropna()

            if len(train_data) < 30:
                print(f"  Skipping {indicator}: insufficient data")
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data[indicator].values
            train_lats = train_data['Latitude'].values
            train_lons = train_data['Longitude'].values
            train_dates = province_data.loc[train_data.index, 'Date'].values
            train_times = [pd.Timestamp(d) for d in train_dates]

            # Initialize and fit GTWR model
            model = GTWR(spatial_bandwidth=0.5, temporal_bandwidth=30,
                        spatial_weight=0.4, temporal_weight=0.6)

            model.fit(X_train, y_train, train_lats, train_lons, train_times)

            # Prepare prediction data
            # Use recent average WQI for future predictions
            recent_wqi = province_data['Water_Quality_Index'].tail(10).mean()

            X_pred = np.array([[mean_lat, mean_lon, recent_wqi]] * forecast_days)
            pred_lats = np.array([mean_lat] * forecast_days)
            pred_lons = np.array([mean_lon] * forecast_days)
            pred_times = [pd.Timestamp(d) for d in future_dates]

            # Make predictions
            predictions = model.predict(X_pred, pred_lats, pred_lons, pred_times)

            # Store results
            results[indicator] = {
                'predictions': predictions,
                'dates': future_dates,
                'historical_mean': y_train.mean(),
                'historical_std': y_train.std(),
                'recent_trend': y_train[-10:].mean() - y_train[-30:-10].mean()
            }

            print(f"  ✓ Predicted mean: {predictions.mean():.2f} (Historical: {y_train.mean():.2f})")

        except Exception as e:
            print(f"  ✗ Error predicting {indicator}: {str(e)}")
            continue

    return results


def visualize_predictions(province, results, output_dir):
    """
    Visualize prediction results for a province
    可视化预测结果
    """
    if not results:
        return

    # Create figure with subplots
    n_indicators = len(results)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_indicators > 1 else [axes]

    for idx, (indicator, data) in enumerate(results.items()):
        ax = axes[idx]

        dates = data['dates']
        predictions = data['predictions']

        # Plot predictions
        ax.plot(dates, predictions, marker='o', linestyle='-',
               linewidth=2, markersize=4, label='Predicted', color='#e74c3c')

        # Add confidence band (±1 std)
        std = data['historical_std']
        ax.fill_between(dates, predictions - std, predictions + std,
                        alpha=0.2, color='#e74c3c', label='±1 STD')

        # Add horizontal line for historical mean
        ax.axhline(data['historical_mean'], color='#3498db',
                  linestyle='--', linewidth=1.5, label='Historical Mean')

        # Formatting
        ax.set_title(indicator.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide empty subplots
    for idx in range(n_indicators, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Water Quality Indicator Predictions - {province}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    filename = f'{output_dir}/{province}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved visualization: {filename}")


def generate_prediction_report(all_results, output_dir):
    """
    Generate comprehensive prediction report
    生成综合预测报告
    """
    report_path = f'{output_dir}/prediction_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GTWR WATER QUALITY PREDICTION REPORT\n")
        f.write("时空地理加权回归水质预测报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Provinces Analyzed: {len(all_results)}\n")
        f.write(f"Forecast Period: 30 days\n\n")

        f.write("-"*80 + "\n\n")

        for province, results in all_results.items():
            if not results:
                continue

            f.write(f"PROVINCE: {province}\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Indicators Predicted: {len(results)}\n\n")

            # Summary statistics for each indicator
            for indicator, data in results.items():
                pred_mean = data['predictions'].mean()
                pred_std = data['predictions'].std()
                hist_mean = data['historical_mean']
                trend = data['recent_trend']

                f.write(f"\n{indicator}:\n")
                f.write(f"  Historical Mean: {hist_mean:.3f}\n")
                f.write(f"  Predicted Mean: {pred_mean:.3f}\n")
                f.write(f"  Predicted Std: {pred_std:.3f}\n")
                f.write(f"  Recent Trend: {trend:+.3f}\n")
                f.write(f"  Change from Historical: {((pred_mean - hist_mean) / hist_mean * 100):+.2f}%\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\nMODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write("Model: Geographically and Temporally Weighted Regression (GTWR)\n")
        f.write("Spatial Bandwidth: 0.5 degrees\n")
        f.write("Temporal Bandwidth: 30 days\n")
        f.write("Spatial Weight: 0.4\n")
        f.write("Temporal Weight: 0.6\n")
        f.write("\n" + "="*80 + "\n")

    print(f"\n✓ Generated prediction report: {report_path}")


def main():
    """
    Main execution function
    主执行函数
    """
    print("="*80)
    print("GTWR WATER QUALITY PREDICTION SYSTEM")
    print("时空地理加权回归水质预测系统")
    print("="*80)

    # Load data
    print("\n[1/4] Loading preprocessed data...")
    df = load_and_prepare_data('preprocessed_data.csv')
    print(f"✓ Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

    # Get unique provinces
    provinces = df['Province'].unique()
    print(f"✓ Found {len(provinces)} provinces: {', '.join(provinces)}")

    # Predict for each province
    print("\n[2/4] Running GTWR predictions for each province...")
    all_results = {}

    for province in provinces:
        results = predict_province_indicators(df, province, forecast_days=30)
        if results:
            all_results[province] = results

    # Visualize results
    print("\n[3/4] Generating visualizations...")
    for province, results in all_results.items():
        visualize_predictions(province, results, OUTPUT_DIR)

    # Generate report
    print("\n[4/4] Generating prediction report...")
    generate_prediction_report(all_results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"  - Province-specific prediction charts: {len(all_results)} files")
    print(f"  - Comprehensive report: prediction_report.txt")
    print("\n✓ All predictions completed successfully!")


if __name__ == "__main__":
    main()
