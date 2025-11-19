import pandas as pd

df = pd.read_csv('C:/Users/YY/Documents/china-water-quality-dataset-analysis/preprocessed_data_feature_eng.csv')

features_to_drop = [
    'Province', 'City', 'Monitoring_Station', 'Date', 'Remarks',
    'Water_Quality_Index',
    'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L',
    'Pollution_Level'
]

df_model = df.drop(columns=features_to_drop)

print('Feature columns:')
for c in df_model.columns:
    print(f'  {c}')
print(f'\nTotal features: {len(df_model.columns)}')
