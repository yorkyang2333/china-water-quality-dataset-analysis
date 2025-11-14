import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Read data
df = pd.read_csv('preprocessed_data_feature_eng.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Separate 2023 and 2024
df_2023 = df[df['Date'].dt.year == 2023].copy()
df_2024 = df[df['Date'].dt.year == 2024].copy()

print("[ANALYSIS] Water Quality Dataset Deep Dive")
print("=" * 80)
print()

# Task 2: Data Distribution Analysis
print("[TASK 2] TEMPORAL DISTRIBUTION ANALYSIS")
print("-" * 80)

# Select representative indicators
indicators = ['Water_Temperature_C', 'COD_mg_L']

for indicator in indicators:
    print(f"\n<<< {indicator} >>>")
    print()
    
    # Remove missing values
    data = df[df[indicator].notna()][indicator].values
    
    # Temporal statistics
    print(f"[OK] Overall Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std:  {np.std(data):.4f}")
    print(f"  Min:  {np.min(data):.4f}")
    print(f"  Max:  {np.max(data):.4f}")
    print()
    
    # Time interval analysis
    df_sorted = df.dropna(subset=[indicator]).sort_values('Date').reset_index(drop=True)
    time_diffs = df_sorted['Date'].diff().dt.days.dropna().values
    
    print(f"[OK] Sampling Interval (days between consecutive measurements):")
    print(f"  Min interval:  {np.min(time_diffs):.0f} days")
    print(f"  Max interval:  {np.max(time_diffs):.0f} days")
    print(f"  Mean interval: {np.mean(time_diffs):.2f} days")
    print(f"  Std interval:  {np.std(time_diffs):.2f} days")
    print(f"  Coefficient of Variation (CV): {np.std(time_diffs)/np.mean(time_diffs):.2f}")
    print()
    
    # Data distribution shape
    first_diff = np.diff(data)
    relative_change = np.abs(first_diff) / (np.abs(data[:-1]) + 1e-6)
    
    print(f"[OK] Volatility Analysis:")
    print(f"  Mean absolute 1st difference: {np.mean(np.abs(first_diff)):.4f}")
    print(f"  Mean relative change: {np.mean(relative_change):.4f}")
    print(f"  Std of 1st difference: {np.std(first_diff):.4f}")
    print()
    
    # Autocorrelation (manual calculation)
    def calc_acf(x, lag):
        c0 = np.mean((x - np.mean(x)) ** 2)
        c_lag = np.mean((x[:-lag] - np.mean(x)) * (x[lag:] - np.mean(x)))
        return c_lag / c0
    
    print(f"[OK] Autocorrelation:")
    for lag in [1, 7, 30]:
        if lag < len(data):
            acf_val = calc_acf(data, lag)
            print(f"  ACF(lag={lag}): {acf_val:.4f}")
    print()

# Task 3: 2023 vs 2024 Comparison
print()
print("[TASK 3] 2023 vs 2024 DISTRIBUTION SHIFT")
print("-" * 80)
print()

print(f"[OK] Sample Counts:")
print(f"  2023: {len(df_2023)} samples")
print(f"  2024: {len(df_2024)} samples")
print(f"  Ratio (2024/2023): {len(df_2024)/len(df_2023):.2%}")
print()

for indicator in indicators:
    print(f"<<< {indicator} >>>")
    
    data_2023 = df_2023[df_2023[indicator].notna()][indicator].values
    data_2024 = df_2024[df_2024[indicator].notna()][indicator].values
    
    print(f"  2023: N={len(data_2023)}, mean={np.mean(data_2023):.4f}, std={np.std(data_2023):.4f}")
    print(f"  2024: N={len(data_2024)}, mean={np.mean(data_2024):.4f}, std={np.std(data_2024):.4f}")
    
    # Calculate Kolmogorov-Smirnov statistic manually (simple version)
    if len(data_2023) > 0 and len(data_2024) > 0:
        # Sort data
        sorted_2023 = np.sort(data_2023)
        sorted_2024 = np.sort(data_2024)
        
        # CDF values
        def compute_ks(x, y):
            n, m = len(x), len(y)
            x_vals = np.concatenate([x, y])
            x_vals = np.unique(np.sort(x_vals))
            
            d_max = 0
            for val in x_vals:
                cdf_x = np.sum(x <= val) / n
                cdf_y = np.sum(y <= val) / m
                d_max = max(d_max, abs(cdf_x - cdf_y))
            return d_max
        
        ks_stat = compute_ks(data_2023, data_2024)
        print(f"  KS-statistic (distribution shift): {ks_stat:.4f}")
        print(f"  -> Significant shift if KS > ~0.05-0.1")
    print()

# Task 4: Per-Province Analysis
print()
print("[TASK 4] PROVINCE-LEVEL TEMPORAL PATTERNS")
print("-" * 80)
print()

provinces = df['Province'].unique()
print(f"[OK] Provinces: {sorted(provinces)}")
print()

for province in ['Beijing', 'Guangdong', 'Sichuan', 'Zhejiang']:
    df_prov = df[df['Province'] == province]
    df_prov_2023 = df_prov[df_prov['Date'].dt.year == 2023]
    df_prov_2024 = df_prov[df_prov['Date'].dt.year == 2024]
    
    print(f"<<< {province} >>>")
    print(f"  2023: {len(df_prov_2023)} samples")
    print(f"  2024: {len(df_prov_2024)} samples")
    
    # Check data availability patterns
    if len(df_prov_2023) > 0:
        date_min_2023 = df_prov_2023['Date'].min()
        date_max_2023 = df_prov_2023['Date'].max()
        print(f"  2023 date range: {date_min_2023.date()} to {date_max_2023.date()}")
    
    if len(df_prov_2024) > 0:
        date_min_2024 = df_prov_2024['Date'].min()
        date_max_2024 = df_prov_2024['Date'].max()
        print(f"  2024 date range: {date_min_2024.date()} to {date_max_2024.date()}")
    print()

print("\n" + "=" * 80)
print("[DONE] Analysis complete")
