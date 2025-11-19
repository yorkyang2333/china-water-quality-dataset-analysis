import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# 加载模型
model_path = Path("C:/Users/YY/Documents/china-water-quality-dataset-analysis/Outputs/SHAP_Analysis_20251119_180422/LightGBM_model.pkl")

with open(model_path, 'rb') as f:
    model, label_encoder, feature_columns = pickle.load(f)

print(f"feature_columns type: {type(feature_columns)}")
print(f"feature_columns: {feature_columns}")
print(f"Number of features: {len(feature_columns) if hasattr(feature_columns, '__len__') else 'N/A'}")

# 加载数据
df = pd.read_csv("C:/Users/YY/Documents/china-water-quality-dataset-analysis/preprocessed_data_feature_eng.csv")
X = df[feature_columns]

print(f"\nX.columns type: {type(X.columns)}")
print(f"X.columns: {X.columns.tolist()}")

# 测试SHAP
import shap
from sklearn.model_selection import train_test_split

y = df['Pollution_Level']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:10])  # 只计算前10个样本

print(f"\nshap_values type: {type(shap_values)}")
print(f"is list: {isinstance(shap_values, list)}")

if isinstance(shap_values, list):
    print(f"shap_values length: {len(shap_values)}")
    print(f"shap_values[0] shape: {shap_values[0].shape}")
    mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
else:
    print(f"shap_values shape: {shap_values.shape}")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

print(f"\nmean_abs_shap type: {type(mean_abs_shap)}")
print(f"mean_abs_shap shape: {mean_abs_shap.shape}")
print(f"mean_abs_shap dtype: {mean_abs_shap.dtype}")

top_indices_array = np.argsort(mean_abs_shap)[::-1][:5]
print(f"\ntop_indices_array type: {type(top_indices_array)}")
print(f"top_indices_array shape: {top_indices_array.shape}")
print(f"top_indices_array: {top_indices_array}")
print(f"top_indices_array[0] type: {type(top_indices_array[0])}")
print(f"top_indices_array[0]: {top_indices_array[0]}")
