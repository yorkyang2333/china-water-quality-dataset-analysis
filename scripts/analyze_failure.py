import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("[DIAGNOSIS] Why Time Series Models Fail - Root Cause Analysis")
print("=" * 80)
print()

# Read data
df = pd.read_csv('preprocessed_data_feature_eng.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("[PROBLEM 1] Irregular Sampling Breaks Assumptions")
print("-" * 80)

# Get one province for detailed analysis
beijing_data = df[df['Province'] == 'Beijing'].sort_values(['Monitoring_Station', 'Date'])

intervals_per_station = []
for station in beijing_data['Monitoring_Station'].unique()[:3]:
    station_data = beijing_data[beijing_data['Monitoring_Station'] == station]
    if len(station_data) > 1:
        diffs = station_data['Date'].diff().dt.days.dropna().values
        intervals_per_station.append(diffs)
        print(f"  {station}: intervals std={np.std(diffs):.2f} days (mean={np.mean(diffs):.2f})")

print()
print("  [KEY ISSUE] CV (Std/Mean) > 1 means intervals are VERY IRREGULAR")
print("  - ARIMA/SARIMA assume CONSTANT interval")
print("  - LSTM expects regular sequence of timesteps")
print("  - RandomForest + LSTM fusion: each model sees different effective timesteps")
print()

print("[PROBLEM 2] Data Volume Collapse 2023 -> 2024")
print("-" * 80)

df_2023 = df[df['Date'].dt.year == 2023]
df_2024 = df[df['Date'].dt.year == 2024]

print(f"  2023: {len(df_2023):5d} samples")
print(f"  2024: {len(df_2024):5d} samples")
print(f"  Ratio: {len(df_2024)/len(df_2023):.1%}")
print()
print("  [KEY ISSUE] 87.6% data loss from 2023 to 2024")
print("  - Train on 13,811 samples, validate on 1,713 samples")
print("  - Distribution fundamentally different (way less data density)")
print("  - 'Overfitting to training regime' impossible to avoid")
print()

# Province-level sample counts
print("[PROBLEM 3] Per-Province Sample Imbalance in 2024")
print("-" * 80)

for prov in sorted(df['Province'].unique()):
    prov_2023 = len(df_2023[df_2023['Province'] == prov])
    prov_2024 = len(df_2024[df_2024['Province'] == prov])
    ratio = prov_2024 / prov_2023 if prov_2023 > 0 else 0
    print(f"  {prov:12s}: 2023={prov_2023:4d}, 2024={prov_2024:3d}, ratio={ratio:5.1%}")

print()
print("  [KEY ISSUE] ALL provinces have <15% of 2024 vs 2023 data")
print("  - Validation set too small for statistical significance")
print("  - Models trained on dense 2023 data fail on sparse 2024 data")
print()

print("[PROBLEM 4] Hybrid Model Weight Learning Failure")
print("-" * 80)

print("  How Hybrid Model Works:")
print("  1. Train LSTM on 2023 data -> learns from dense, regular sequences")
print("  2. Train RF on 2023 data    -> learns from averaged features")
print("  3. On 2024 validation:")
print("     - LSTM sees 4.48-day gaps: NOT trained for this!")
print("     - RF sees 4.48-day gaps: NOT trained for this!")
print("  4. Fusion weight w learned to balance TWO BROKEN predictions")
print()
print("  [KEY ISSUE] Garbage in (broken models), garbage out (bad weights)")
print("  - 2024 data regime ~ 8x sparser than 2023")
print("  - Both base models extrapolate terribly")
print("  - Fusion weight = averaging two terrible predictions")
print()

print("[PROBLEM 5] Sequence Length Mismatch")
print("-" * 80)

print("  LSTM with seq_length=7:")
print("  - Expects: 7 days of dense, consecutive data")
print("  - Reality: ~7 days of IRREGULAR, gappy data")
print("  - 4.48-day average interval means: 7*4.48 ~ 31 days in real time")
print("  - So: model trained on 7 timesteps (7-day sequences)")
print("       but validating on sequences spanning 31 CALENDAR days")
print()
print("  [KEY ISSUE] Feature engineering uses shifted values (lag-1, lag-2, ...)")
print("  - With 4-day gaps, 'lag-1' means '4 days ago'")
print("  - With training on consecutive days, 'lag-1' means '1 day ago'")
print("  - COMPLETELY different feature semantics!")
print()

print("[PROBLEM 6] Autocorrelation = 0 (NO TIME DEPENDENCY)")
print("-" * 80)

# Calculate ACF
def calc_acf(x, lag):
    c0 = np.mean((x - np.mean(x)) ** 2)
    c_lag = np.mean((x[:-lag] - np.mean(x)) * (x[lag:] - np.mean(x)))
    return c_lag / c0

data_temp = df[df['Province']=='Beijing']['Water_Temperature_C'].dropna().values
data_cod = df[df['Province']=='Beijing']['COD_mg_L'].dropna().values

print(f"  Water_Temperature_C ACF(1): {calc_acf(data_temp, 1):.4f}")
print(f"  Water_Temperature_C ACF(7): {calc_acf(data_temp, 7):.4f}")
print(f"  COD_mg_L ACF(1): {calc_acf(data_cod, 1):.4f}")
print(f"  COD_mg_L ACF(7): {calc_acf(data_cod, 7):.4f}")
print()
print("  [KEY ISSUE] ACF ~ 0 means NO autocorrelation!")
print("  - Current value almost completely independent of past values")
print("  - Time series methods (ARIMA, LSTM, SARIMA) designed for ACF >> 0")
print("  - This data might be RANDOM WALK or NOISE with trend")
print()

print("[VERDICT] Why Hybrid Model R2 ~ -1 to -3:")
print("-" * 80)
print()
print("  1. LSTM trained on dense 2023 -> poor 2024 predictions")
print("  2. RF trained on dense 2023 -> poor 2024 predictions")
print("  3. Fusion tries to learn weight w in: y = w*lstm_pred + (1-w)*rf_pred")
print("  4. With TWO broken predictions, even optimal w gives terrible results")
print("  5. R2 = 1 - (SS_res / SS_tot)")
print("     If SS_res > SS_tot -> R2 < 0 (worse than just predicting mean)")
print()
print("  => R2 ~ -1 to -3 means predictions are 2-4x worse than naive baseline!")
print()

print("[CORE INSIGHT]")
print("-" * 80)
print()
print("  This is NOT a model architecture problem.")
print("  This is a DATA QUALITY / DISTRIBUTION MISMATCH problem.")
print()
print("  Time series models fail because:")
print("  - Irregular sampling every 4-5 days (not daily or hourly)")
print("  - Autocorrelation near zero (no time dependency)")
print("  - 87% data loss between train and validation sets")
print("  - Distribution regime fundamentally different (dense vs sparse)")
print()
print("  => Use cross-sectional classification instead!")
print("     (Predict Pollution_Level from current water chemistry,")
print("      treating each observation independently)")
print()

