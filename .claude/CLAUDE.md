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
├── eda_map.py                            # ⭐ 新增：省级交互式地图可视化脚本
├── eda_feature_engineering.py            # EDA及特征工程脚本
├── classification_random_forest.py       # 随机森林分类模型脚本
├── classification_xgboost.py             # XGBoost分类模型脚本
├── classification_svm.py                 # SVM分类模型脚本
├── classification_logistic_regression.py # 逻辑回归分类模型脚本
├── requirements.txt                      # Python依赖包列表
│
├── timeseries_models_province/           # ⭐ 时间序列模型 - 按省份建模 (不推荐)
│   └── ...
│
├── timeseries_models_station/            # 时间序列模型 - 按监测站建模 (不推荐)
│   └── ...
│
├── backup_prediction_attempts_crosssectional/  # 横截面预测尝试（备份）
│   └── ...
│
└── Outputs/                              # 所有模型的输出结果
    ├── EDA/                              # 探索性数据分析结果
    ├── EDA_Map/                          # ⭐ 新增：交互式地图输出
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

### 2. 生成交互式地图
```bash
# 此脚本会生成一个多图层的省级水质地图
python eda_map.py
# 输出: Outputs/EDA_Map/province_level_water_quality_map.html
```

### 3. 运行分类模型
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

## 📊 模型评估总结 (Classification Models Summary)

我们针对 `Pollution_Level` 的预测任务，对四个不同的分类模型进行了评估。所有模型均使用相同的特征集和测试集。

| 模型 | 准确率 (Accuracy) | 核心结论 |
| :--- | :--- | :--- |
| **XGBoost** | **~99.58%** | **性能最佳**。在所有类别上都表现出极高的精确率和召回率，是本项目最推荐的预测模型。 |
| **随机森林** | **~99.55%** | 性能与XGBoost非常接近，同样表现出色。其内置的特征重要性分析也很有价值。 |
| **支持向量机 (SVM)** | **~98.16%** | 表现良好，但准确率略低于基于树的集成模型。需要通过精细调参来进一步提升性能。 |
| **逻辑回归** | **~96.17%** | 作为线性基准模型，其准确率说明了问题的复杂性和非线性特征的重要性。 |

**总结**: 对于此数据集，**XGBoost** 和 **随机森林** 是预测水质污染等级的最佳选择，准确率均超过99.5%。

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

---

**创建日期**: 2025-01-05
**最后更新**: 2025-10-15
**维护者**: Gemini Agent
**版本**: 2.2 (新增地图可视化功能)
