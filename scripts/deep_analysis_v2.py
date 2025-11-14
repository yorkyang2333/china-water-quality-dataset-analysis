import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Read data
df = pd.read_csv('preprocessed_data_feature_eng.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

print("[ANALYSIS] Critical: Time Interval Analysis - Station-Level")
print("=" * 80)
print()

# Analyze by station
stations = df.groupby(['Province', 'Monitoring_Station'])

print("[TASK] Station-level time intervals (first 10 stations)")
print("-" * 80)
print()

count = 0
all_intervals = []

for (prov, station), group in stations:
    if count >= 10:
        break
    
    group_sorted = group.sort_values('Date').reset_index(drop=True)
    
    if len(group_sorted) > 1:
        time_diffs = group_sorted['Date'].diff().dt.days.dropna().values
        
        if len(time_diffs) > 0:
            all_intervals.extend(time_diffs)
            
            print(f"[{count+1}] {prov} - {station}")
            print(f"    N={len(group_sorted)}, intervals: min={np.min(time_diffs):.0f}, max={np.max(time_diffs):.0f}, mean={np.mean(time_diffs):.2f}, std={np.std(time_diffs):.2f}")
    
    count += 1

print()
print("[CRITICAL FINDING] Global Time Interval Statistics")
print("-" * 80)

if len(all_intervals) > 0:
    all_intervals = np.array(all_intervals)
    print(f"Total intervals analyzed: {len(all_intervals)}")
    print(f"Min: {np.min(all_intervals):.0f} days")
    print(f"Max: {np.max(all_intervals):.0f} days")
    print(f"Mean: {np.mean(all_intervals):.2f} days")
    print(f"Median: {np.median(all_intervals):.2f} days")
    print(f"Std: {np.std(all_intervals):.2f} days")
    print(f"P25: {np.percentile(all_intervals, 25):.2f} days")
    print(f"P75: {np.percentile(all_intervals, 75):.2f} days")
    print(f"P95: {np.percentile(all_intervals, 95):.2f} days")
    print()
    
    # Distribution of intervals
    bins = [0, 1, 2, 3, 5, 7, 14, 30, 60, 100, 500]
    hist, _ = np.histogram(all_intervals, bins=bins)
    print("[INTERVAL DISTRIBUTION]")
    for i, count_val in enumerate(hist):
        if i < len(bins) - 1:
            pct = 100.0 * count_val / len(all_intervals)
            print(f"  {bins[i]:3.0f}-{bins[i+1]:3.0f} days: {count_val:6d} ({pct:5.1f}%)")

print()
print("[ANALYSIS] Data completeness across 2023 and 2024")
print("-" * 80)

df_2023 = df[df['Date'].dt.year == 2023]
df_2024 = df[df['Date'].dt.year == 2024]

print(f"2023 total records: {len(df_2023)}")
print(f"2024 total records: {len(df_2024)}")
print(f"2024/2023 ratio: {len(df_2024)/len(df_2023):.1%}")
print()

# Check monthly distribution in 2024
print("[2024 MONTHLY DISTRIBUTION]")
df_2024['Month'] = df_2024['Date'].dt.month
for month in range(1, 13):
    month_data = df_2024[df_2024['Month'] == month]
    if len(month_data) > 0:
        print(f"  Month {month:2d}: {len(month_data):4d} records")
    else:
        print(f"  Month {month:2d}: ---- (NO DATA)")

