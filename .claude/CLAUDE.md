# 水质污染数据分析项目 - Gemini Agent 配置

## 项目背景
这是一个水质污染数据分析项目，包含数据预处理、探索性数据分析(EDA)和机器学习预测模型。项目涵盖了时间序列预测和横截面预测两大类方法。

## 📁 项目结构

```
china-water-quality-dataset-analysis/
├── preprocessed_data.csv                 # 预处理后的数据
├── preprocessed_data_feature_eng.csv     # 带有新工程特征的数据
├── preprocess.py                         # 数据预处理脚本
├── eda.py                                # 探索性数据分析脚本
├── eda_map.py                            # ⭐ 省级交互式地图可视化脚本
├── eda_feature_engineering.py            # EDA及特征工程脚本
├── timeseries_visualization_eda.py       # ⭐⭐ NEW: 时间序列可视化与EDA
├── classification_random_forest.py       # 随机森林分类模型脚本
├── classification_xgboost.py             # XGBoost分类模型脚本
├── classification_svm.py                 # SVM分类模型脚本
├── classification_logistic_regression.py # 逻辑回归分类模型脚本
├── timeseries_station_prediction_revised.py  # ⭐⭐ NEW: 时间序列预测(改进版)
├── requirements.txt                      # Python依赖包列表
│
├── timeseries_models_province/           # 时间序列模型 - 按省份建模 (不推荐)
│   └── ...
│
├── timeseries_models_station/            # 时间序列模型 - 按监测站建模 (不推荐)
│   └── ...
│
├── timeseries_models_station_revised/    # ⭐⭐ NEW: 改进版时间序列模型 - Station级别
│   ├── README.md
│   ├── arima_station_revised.py
│   ├── lstm_station_revised_auto_gpu.py  # ⭐ 支持CUDA/MPS自动GPU加速
│   └── ...
│
├── timeseries_models_province_revised/   # ⭐⭐ NEW: 改进版时间序列模型 - Province级别
│   ├── README.md
│   ├── arima_province_revised.py
│   ├── lstm_province_revised_auto_gpu.py # ⭐ 支持CUDA/MPS自动GPU加速
│   └── ...
│
├── backup_prediction_attempts_crosssectional/  # 横截面预测尝试（备份）
│   └── ...
│
└── Outputs/                              # 所有模型的输出结果
    ├── EDA/                              # 探索性数据分析结果
    ├── EDA_Map/                          # 交互式地图输出
    ├── Timeseries_EDA/                   # ⭐⭐ NEW: 时间序列EDA输出
    ├── Feature_Engineering_EDA/          # 特征工程分析结果
    ├── Classification_RandomForest/      # 随机森林模型输出
    ├── Classification_XGBoost/           # XGBoost模型输出
    ├── Classification_SVM/               # SVM模型输出
    ├── Classification_LogisticRegression/  # 逻辑回归模型输出
    └── ...                               # 其他模型输出
```

## 🎯 核心数据集特点（重要！）

### ✅ 优势
- **数据量充足**：共180个监测站，超过15000条观测记录。
- **覆盖范围广**：10个省份，多个城市。
- **指标齐全**：16个水质指标 + 3个新工程特征。

### ⚠️ 关键问题：时间间隔极不规则
**这是影响模型选择的最重要因素！** 数据集是 **不规则抽样**，违反了大多数时间序列模型的等间隔采样假设，因此不适合使用时间序列方法。

### ✨ 新创建的组合特征 (Engineered Features)
为了增强模型性能，我们通过 `eda_feature_engineering.py` 脚本创建了以下新特征，并保存在 `preprocessed_data_feature_eng.csv` 中：
- **`Total_Heavy_Metals_ug_L`**: 铅、镉、汞的总和，综合反映重金属污染。
- **`COD_BOD_Ratio`**: COD与BOD的比值，反映污水中难生物降解有机物的比例。
- **`NP_Ratio`**: 总氮与总磷的比值，评估水体富营养化风险。

## ✅ 推荐方法（有效）

### 1. 横截面分类预测（最推荐）
我们已经成功构建了以下模型，用于预测 `Pollution_Level`（污染等级）：
- **随机森林 (Random Forest)**: 效果强大且稳定，可解释性好。
- **XGBoost**: 性能卓越，通常比随机森林更精准。
- **支持向量机 (SVM)**: 在高维空间中表现良好，提供了不同的决策边界思路。
- **逻辑回归 (Logistic Regression)**: 作为性能基准，用于衡量复杂模型的提升程度。

### 2. 多指标回归
- **方法**: 利用相关性强的多个水质指标来预测另一个目标指标（如 `Water_Quality_Index`）。
- **模型**: 可使用线性回归、随机森林回归、XGBoost回归等。

## 🗺️ 地图可视化 (Map Visualization)
新增的 `eda_map.py` 脚本提供了一个强大的交互式地理可视化工具，用于探索各省份的水质状况。

- **功能**: 生成一个以省份为单位的多图层交互式地图。
- **核心特性**:
  - **分层设色**: 根据不同水质指标（如水质指数、COD、总氮等）的平均值对省份进行颜色渲染。
  - **图层切换**: 用户可以在地图控件中方便地切换不同的指标图层，进行对比分析。
  - **详细信息**: 点击每个省份的中心标记，可以弹出窗口显示该地区详细的水质统计数据。
- **技术栈**: 使用 `folium` 库生成 HTML 地图，并从网络实时获取 GeoJSON 地理数据。

## 🚀 快速开始

### 1. 运行探索性数据分析与特征工程
```bash
# 此脚本会生成新的组合特征并进行初步分析
python eda_feature_engineering.py
# 输出: Outputs/Feature_Engineering_EDA/ 和 preprocessed_data_feature_eng.csv
```

### 2. ⭐⭐ NEW: 运行时间序列可视化EDA
```bash
# 为所有180个监测站生成时间序列折线图，分析采样间隔
python timeseries_visualization_eda.py
# 输出: Outputs/Timeseries_EDA/ (180个站点图表 + 汇总分析)
```

### 3. 生成交互式地图
```bash
# 此脚本会生成一个多图层的省级水质地图
python eda_map.py
# 输出: Outputs/EDA_Map/province_level_water_quality_map.html
```

### 4. 运行分类模型（推荐）
```bash
# 运行随机森林模型
python classification_random_forest.py

# 运行XGBoost模型
python classification_xgboost.py

# 运行SVM模型
python classification_svm.py

# 运行逻辑回归模型
python classification_logistic_regression.py

# 所有模型的报告、混淆矩阵和特征重要性图表都会保存在Outputs/下的对应文件夹中。
```

### 5. ⭐⭐ NEW: 运行改进版时间序列模型（实验性）
```bash
# 传统机器学习时间序列预测（RandomForest/Ridge，自动GPU加速）
python timeseries_station_prediction_revised.py

# ARIMA模型（改进版：2023训练，2024验证）
python timeseries_models_station_revised/arima_station_revised.py

# LSTM模型（改进版 + 自动GPU加速：CUDA/MPS/CPU）
python timeseries_models_station_revised/lstm_station_revised_auto_gpu.py

# 注意：尽管进行了改进，时间序列方法仍然不如横截面分类方法
```

## 📊 模型评估总结 (Classification Models Summary)

我们针对 `Pollution_Level` 的预测任务，对四个不同的分类模型进行了评估。所有模型均使用相同的特征集和测试集。

| 模型 | 准确率 (Accuracy) | 核心结论 |
| :--- | :--- | :--- |
| **XGBoost** | **~99.58%** | **性能最佳**。在所有类别上都表现出极高的精确率和召回率，是本项目最推荐的预测模型。 |
| **随机森林** | **~99.55%** | 性能与XGBoost非常接近，同样表现出色。其内置的特征重要性分析也很有价值。 |
| **支持向量机 (SVM)** | **~98.16%** | 表现良好，但准确率略低于基于树的集成模型。需要通过精细调参来进一步提升性能。 |
| **逻辑回归** | **~96.17%** | 作为线性基准模型，其准确率说明了问题的复杂性和非线性特征的重要性。 |

**总结**: 对于此数据集，**XGBoost** 和 **随机森林** 是预测水质污染等级的最佳选择，准确率均超过99.5%。

## ⭐⭐ 时间序列模型改进版 (Revised Time Series Models) - NEW!

### 背景
通过时间序列EDA分析发现，2024年的数据明显少于2023年，这可能是导致原始时间序列模型准确率极低的主要原因。因此，我们创建了改进版本：

### 关键改进
1. **数据分割策略**：
   - ❌ 原版本：按时间顺序80/20分割
   - ✅ 改进版：仅用2023年数据训练，2024年数据验证

2. **可视化输出**：
   - ❌ 原版本：散点图（Actual vs Predicted）
   - ✅ 改进版：折线图（更清晰展示时间趋势）

3. **GPU加速支持**（LSTM模型）：
   - ✅ 自动检测CUDA GPU（NVIDIA）
   - ✅ 自动检测MPS（Apple Silicon）
   - ✅ 自动回退到CPU（通用）

### 包含的模型
#### Station级别（按监测站建模）
- `timeseries_station_prediction_revised.py` - RandomForest/Ridge
- `arima_station_revised.py` - ARIMA模型
- `lstm_station_revised_auto_gpu.py` - LSTM + 自动GPU加速

#### Province级别（按省份建模）
- `arima_province_revised.py` - ARIMA模型
- `sarima_province_revised.py` - SARIMA模型
- `prophet_province_revised.py` - Prophet模型
- `lstm_province_revised_auto_gpu.py` - LSTM + 自动GPU加速

### ⚠️ 重要提醒
虽然进行了针对性改进，但由于**数据采样间隔极不规则**（标准差16天，最大253天），时间序列方法仍然不适合此数据集。改进版本主要用于：
- ✓ 验证假设
- ✓ 对比实验
- ✓ 学习研究
- ✗ **不建议作为主要预测方法**

**推荐继续使用横截面分类方法（XGBoost、Random Forest）！**

## 🔧 环境配置

### Python依赖
```bash
pip install -r requirements.txt
```
如果缺少 `xgboost`，请单独安装:
```bash
pip install xgboost
```

### 主要库
- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn, xgboost
- **可视化**: matplotlib, seaborn, folium

## 📝 版本历史

- **v2.3** (2025-10-25) - Claude Code
  - ⭐⭐ 新增时间序列可视化EDA
  - ⭐⭐ 新增改进版时间序列模型（2023训练/2024验证）
  - ⭐⭐ 新增LSTM自动GPU加速支持（CUDA/MPS）
  - 批量修改10个时间序列模型
  - 创建完整的技术文档和README

- **v2.2** (2025-10-15) - Gemini Agent
  - 新增地图可视化功能

- **v2.0** (2025-01-05) - Gemini Agent
  - 初始版本

---

**创建日期**: 2025-01-05
**最后更新**: 2025-10-25
**维护者**: Claude Code (v2.3), Gemini Agent (v2.0-v2.2)
**版本**: 2.3 (新增时间序列改进版 + GPU加速)
