import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = 'Province_Predictions_Improved'
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


def build_prediction_model(province_data, indicator, train_ratio=0.8):
    """
    Build prediction model for a specific indicator with 80/20 train-test split
    使用多种模型预测水质指标

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

    # Add temporal features
    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)

    # Extract temporal features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['Quarter'] = data['Date'].dt.quarter

    # Add lag features (previous values)
    for lag in [1, 7, 30]:
        data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)

    # Add rolling statistics
    for window in [7, 30]:
        data[f'{indicator}_rolling_mean_{window}'] = data[indicator].rolling(window=window, min_periods=1).mean()
        data[f'{indicator}_rolling_std_{window}'] = data[indicator].rolling(window=window, min_periods=1).std().fillna(0)

    # Prepare feature columns
    temporal_features = ['Year', 'Month', 'Day', 'DayOfYear', 'Quarter']
    lag_features = [col for col in data.columns if '_lag_' in col or '_rolling_' in col]

    # Select all available feature indicators that have data
    available_features = [feat for feat in feature_indicators if feat in data.columns]

    feature_cols = temporal_features + available_features + lag_features

    # Prepare data
    required_cols = feature_cols + [indicator]
    model_data = data[required_cols].dropna()

    if len(model_data) < 50:  # Minimum samples required
        return None

    # Split into train and test sets (80/20)
    split_idx = int(len(model_data) * train_ratio)
    train_data = model_data.iloc[:split_idx]
    test_data = model_data.iloc[split_idx:]

    if len(train_data) < 30 or len(test_data) < 5:
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
        # Try multiple models and choose the best one
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }

        best_model = None
        best_r2 = -np.inf
        best_model_name = None
        best_pred = None

        for model_name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Calculate R2
            r2 = r2_score(y_test, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = model_name
                best_pred = y_pred

        # Use best model
        y_pred = best_pred

        # Calculate accuracy metrics
        metrics = calculate_accuracy_metrics(y_test, y_pred)

        if metrics is None:
            return None

        # Get feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:10]

        # Store results
        results = {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': scaler,
            'features': feature_cols,
            'feature_importance': feature_importance,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'train_dates': train_data.index.values,
            'test_dates': test_data.index.values,
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
    Build prediction models for all water quality indicators in a province
    为省份的所有水质指标建立预测模型

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
        if valid_count < 50:
            print(f"  ✗ Insufficient data (only {valid_count} valid samples)")
            continue

        # Build prediction model
        results = build_prediction_model(province_data, indicator, train_ratio=0.8)

        if results:
            province_results[indicator] = results
            metrics = results['metrics']
            print(f"  ✓ Model trained successfully ({results['model_name']})")
            print(f"    Train samples: {results['train_size']} | Test samples: {results['test_size']}")
            print(f"    R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")

            if results['feature_importance']:
                top_features = ', '.join([f[0] for f in results['feature_importance'][:3]])
                print(f"    Top features: {top_features}")
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
        textstr = (f"Model: {result['model_name']}\n"
                  f"R² = {metrics['R2']:.3f}\n"
                  f"RMSE = {metrics['RMSE']:.3f}\n"
                  f"MAE = {metrics['MAE']:.3f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_xlabel('Actual Values', fontsize=10)
        ax.set_ylabel('Predicted Values', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Prediction Models - {province} (80/20 Train-Test Split)',
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
        y_test = result['y_test']
        y_pred = result['y_pred']

        for i in range(len(y_test)):
            all_predictions.append({
                'Province': province,
                'Sample_Index': result['test_dates'][i],
                'Indicator': indicator,
                'Model': result['model_name'],
                'Actual_Value': y_test[i],
                'Predicted_Value': y_pred[i],
                'Error': y_pred[i] - y_test[i],
                'Absolute_Error': abs(y_pred[i] - y_test[i]),
                'Percentage_Error': abs((y_pred[i] - y_test[i]) / (y_test[i] + 1e-10)) * 100
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
        f.write("WATER QUALITY PREDICTION REPORT\n")
        f.write("水质指标预测报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Split: 80% Training / 20% Testing\n")
        f.write(f"Models Used: RandomForest, GradientBoosting, Ridge\n")
        f.write(f"Total Provinces Analyzed: {len(all_province_results)}\n\n")

        # Overall statistics
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_mape = []
        model_counts = {}

        for province, results in all_province_results.items():
            for indicator, result in results.items():
                metrics = result['metrics']
                all_r2.append(metrics['R2'])
                all_rmse.append(metrics['RMSE'])
                all_mae.append(metrics['MAE'])
                all_mape.append(metrics['MAPE'])

                model_name = result['model_name']
                model_counts[model_name] = model_counts.get(model_name, 0) + 1

        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean R² Score: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n")
        f.write(f"Median R² Score: {np.median(all_r2):.4f}\n")
        f.write(f"Mean RMSE: {np.mean(all_rmse):.4f} (±{np.std(all_rmse):.4f})\n")
        f.write(f"Mean MAE: {np.mean(all_mae):.4f} (±{np.std(all_mae):.4f})\n")
        f.write(f"Mean MAPE: {np.mean(all_mape):.2f}% (±{np.std(all_mape):.2f}%)\n")
        f.write(f"Total Indicators Evaluated: {len(all_r2)}\n\n")

        f.write("Model Selection Summary:\n")
        for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model_name}: {count} times ({count/len(all_r2)*100:.1f}%)\n")
        f.write("\n")

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

            f.write(f"{'Indicator':<30} {'Model':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Train':>8} {'Test':>8}\n")
            f.write("-"*80 + "\n")

            for indicator, result in sorted(results.items()):
                metrics = result['metrics']
                f.write(f"{indicator:<30} "
                       f"{result['model_name']:<15} "
                       f"{metrics['R2']:>8.4f} "
                       f"{metrics['RMSE']:>10.4f} "
                       f"{metrics['MAE']:>10.4f} "
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

        f.write("="*80 + "\n")

    print(f"\n✓ Generated comprehensive report: {report_path}")


def main():
    """
    Main execution function
    主执行函数
    """
    print("="*80)
    print("WATER QUALITY PREDICTION SYSTEM - IMPROVED VERSION")
    print("水质指标预测系统 - 改进版")
    print("="*80)

    # Load data
    print("\n[Step 1/5] Loading preprocessed data...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Province', 'Date']).reset_index(drop=True)

    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    provinces = sorted(df['Province'].unique())
    print(f"✓ Found {len(provinces)} provinces: {', '.join(provinces)}")

    # Build prediction models for each province
    print("\n[Step 2/5] Building prediction models for each province...")
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
    print(f"\n✓ Successfully trained {total_indicators} models across {len(all_province_results)} provinces")
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()
