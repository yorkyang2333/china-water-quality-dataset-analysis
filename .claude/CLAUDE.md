# 水质污染数据分析项目 - 项目配置文档

**版本**: 2.8 (重构版)
**最后更新**: 2025-11-22
**维护者**: Claude Code

## 📚 相关文档

- **[CHANGELOG.md](../CHANGELOG.md)** - 完整版本历史（v2.0-v2.7）
- **[API_REFERENCE.md](../docs/API_REFERENCE.md)** - REST API 端点文档
- **[PROJECT_STRUCTURE.md](../docs/project_management/PROJECT_STRUCTURE.md)** - 详细项目结构说明
- **[PAPER_FRAMEWORK.md](../docs/PAPER_FRAMEWORK.md)** - 学术论文框架

---

## 项目背景

这是一个水质污染数据分析项目，包含数据预处理、探索性数据分析(EDA)和机器学习预测模型。项目涵盖了时间序列预测和横截面预测两大类方法。

## 📁 项目结构

```
china-water-quality-dataset-analysis/
│
├── 📊 数据文件
│   ├── 中国水污染监测数据集.csv          # 原始数据集
│   ├── preprocessed_data.csv            # 预处理后的数据
│   └── preprocessed_data_feature_eng.csv # 带有新工程特征的数据
│
├── 🔧 预处理模块 (preprocess/)
│   ├── preprocess.py                    # 数据预处理脚本
│   ├── preprocessed_data.csv            # 预处理后的数据（副本）
│   └── preprocessed_data_feature_eng.csv # 特征工程数据（副本）
│
├── 📈 探索性数据分析 (EDA/)
│   ├── eda.py                           # 基础EDA脚本
│   ├── eda_map.py                       # ⭐ 省级交互式地图可视化
│   ├── eda_feature_engineering.py       # 特征工程分析
│   ├── timeseries_visualization_eda.py  # 时间序列可视化EDA
│   ├── seasonality_analysis_sarima.py   # ⭐⭐ 季节性特征分析
│   ├── bayesian_network_analysis.py     # 贝叶斯网络分析
│   ├── spatial_distribution_analysis.py # 空间分布分析
│   └── temporal_distribution_analysis.py # 时间分布分析
│
├── 🤖 机器学习模型 (models/)
│   │
│   ├── pollution_level_classification/  # ⭐ 污染等级分类模型（推荐）
│   │   ├── 基础模型（7个）:
│   │   │   ├── classification_random_forest.py      # 随机森林
│   │   │   ├── classification_xgboost.py            # XGBoost
│   │   │   ├── classification_svm.py                # 支持向量机
│   │   │   ├── classification_logistic_regression.py # 逻辑回归
│   │   │   ├── classification_catboost.py           # CatBoost
│   │   │   ├── classification_lightgbm.py           # LightGBM
│   │   │   └── classification_knn.py                # K近邻
│   │   │
│   │   └── 高级模型（10个）:
│   │       ├── transformer_classifier.py            # Transformer分类器（v2.7）
│   │       ├── ultimate_ensemble_system.py          # 终极集成系统（v2.7）
│   │       ├── deep_learning_classifier.py          # 深度学习分类器
│   │       ├── enhanced_deep_learning_classifier.py # 增强深度学习
│   │       ├── deep_model_fusion_system.py          # 深度模型融合
│   │       └── ... (其他实验性模型)
│   │
│   ├── indicator_regression/            # ⭐ 横截面回归预测（论文主要方法）
│   │   ├── regression_xgboost.py                # XGBoost回归
│   │   ├── regression_xgboost_clean.py          # 优化版XGBoost回归
│   │   ├── feature_engineering_advanced.py      # 高级特征工程
│   │   └── generate_improvement_comparison.py   # 性能对比生成
│   │
│   ├── timeseries_models_province_revised/ # ⭐⭐⭐ 改进版时间序列 - Province级别
│   │   ├── arima_province_revised.py
│   │   ├── sarima_province_revised.py
│   │   ├── prophet_province_revised.py
│   │   ├── ar_province_revised.py
│   │   ├── lstm_province_revised_auto_gpu.py       # 自动GPU加速
│   │   ├── lstm_province_optimized_auto_gpu.py     # 优化版LSTM
│   │   ├── lstm_rf_hybrid_province.py              # ⭐⭐⭐ LSTM+RF混合模型
│   │   ├── autoformer_province.py                  # Autoformer时序模型（v2.7）
│   │   ├── run_all.py                              # 批量运行脚本
│   │   └── DEPRECATED_ANALYSIS.md                  # ⚠️ 时序方法弃用分析（57KB）
│   │
│   └── archived/                         # 📦 归档模型（不再维护）
│       ├── timeseries_models_station/           # Station级别模型（旧版）
│       ├── timeseries_models_station_revised/   # Station级别模型（改进版）
│       └── backup_prediction_attempts_crosssectional/  # 横截面预测尝试（备份）
│
├── 📊 SHAP可解释性分析 (SHAP_Analysis_20251119_180422/)  # 🆕 重要分析结果
│   ├── Classification_LightGBM/         # 分类模型SHAP分析
│   │   ├── SHAP_REPORT.md              # SHAP分析报告
│   │   └── *.png                       # 可视化图表（28个）
│   └── Regression_XGBoost_Regression_Clean/  # 回归模型SHAP分析
│       ├── SHAP_REPORT.md              # SHAP分析报告
│       └── *.png                       # 可视化图表（27个）
│
├── 📁 文档目录 (docs/)                  # v2.5.2 新增
│   ├── PAPER_FRAMEWORK.md              # 论文框架（31KB）
│   ├── PAPER_MANUSCRIPT_CN.md          # 论文手稿（58KB）
│   ├── SHAP_INTERPRETABILITY_ANALYSIS.md  # SHAP分析总结
│   ├── analysis_reports/               # 分析报告
│   ├── model_experiments/              # 模型实验记录
│   └── project_management/             # 项目管理文档
│       ├── PROJECT_STRUCTURE.md        # 项目结构说明
│       ├── FINAL_DELIVERY_COMPLETE.md  # 论文交付总结
│       └── PROGRESS_REPORT_*.md        # 进度报告
│
├── 🔬 分析脚本 (scripts/)              # v2.5.2 新增
│   ├── generate_paper_figures.py       # 生成论文图表
│   ├── generate_shap_analysis.py       # 生成SHAP分析
│   ├── bayesian_network_analysis.py    # 贝叶斯网络分析
│   ├── auto_select_best_interpretable_model.py  # 自动选择最佳模型
│   └── run_all.py                      # 批量运行脚本
│
├── 🌐 平台演示 (platform_demo/)
│   └── backend/                        # Flask后端服务（⚠️ 注意位置）
│       ├── app.py                      # Flask应用主文件
│       ├── models/                     # 模型实现
│       │   └── lstm_rf_hybrid.py       # 混合模型（用于API）
│       └── data/                       # 数据目录
│
└── 📊 输出目录 (Outputs/)              # 模型输出（动态生成）
    └── Bayesian_Network_Analysis/      # 贝叶斯网络分析输出
```

## 🎯 核心数据集特点（重要！）

### ✅ 优势
- **数据量充足**：共180个监测站，超过15000条观测记录。
- **覆盖范围广**：10个省份，多个城市。
- **指标齐全**：16个水质指标 + 3个新工程特征。

### ⚠️ 关键问题：时间间隔极不规则

**这是影响模型选择的最重要因素！** 数据集是 **不规则抽样**，违反了大多数时间序列模型的等间隔采样假设，因此不适合使用时间序列方法。

**⭐⭐⭐ 重要更新 (2025-11-13)**: 经过系统性分析，**时间序列方法已被正式标记为弃用（DEPRECATED）**。

#### 🚫 时间序列方法状态：已弃用（DEPRECATED）

**弃用原因** (详见 `models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md`):
1. **采样极度不规则** (CV=1.99，远超阈值0.2)
2. **零时间相关性** (ACF≈0，过去无法预测未来)
3. **分布严重漂移** (2024年数据密度仅为2023年的12.4%)

**性能表现**:
- LSTM: R² ≈ -0.1（略差于Naive基线）
- RandomForest时序: R² ≈ -0.05（接近Naive）
- LSTM+RF混合: R² ≈ -0.3（修复后，原为-2.5）

**保留目的**:
- ✅ 作为论文对比基线（展示方法选择的重要性）
- ✅ 失败案例分析参考（学术价值）
- ❌ **不推荐用于实际预测或进一步开发**

**详细分析**: 见 `models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md`（57KB学术报告）

### ✨ 新创建的组合特征 (Engineered Features)

为了增强模型性能，我们通过 `eda_feature_engineering.py` 脚本创建了以下新特征，并保存在 `preprocessed_data_feature_eng.csv` 中：
- **`Total_Heavy_Metals_ug_L`**: 铅、镉、汞的总和，综合反映重金属污染。
- **`COD_BOD_Ratio`**: COD与BOD的比值，反映污水中难生物降解有机物的比例。
- **`NP_Ratio`**: 总氮与总磷的比值，评估水体富营养化风险。

## ✅ 推荐方法（有效）

### 1. 横截面分类预测（最推荐）

我们已经成功构建了以下模型，用于预测 `Pollution_Level`（污染等级）：

**基础模型（7个）**：
- **随机森林 (Random Forest)**: 效果强大且稳定，可解释性好（准确率 ~99.55%）
- **XGBoost**: 性能卓越，最高准确率（~99.58%）
- **CatBoost**: 专门处理分类特征的梯度提升算法，性能优异
- **LightGBM**: 高效的梯度提升框架，训练速度快
- **支持向量机 (SVM)**: 在高维空间中表现良好（~98.16%）
- **K近邻 (KNN)**: 基于实例的学习方法，适合局部模式识别
- **逻辑回归 (Logistic Regression)**: 作为性能基准（~96.17%）

**高级模型（v2.7新增）**：
- **Transformer 分类器**: Multi-Head Self-Attention机制，预期准确率 ~99%+
- **终极集成系统**: 9个基础模型 + 3种集成策略，准确率 99.6%+

### 2. 横截面回归预测（论文主要方法）

- **方法**: 利用相关性强的多个水质指标来预测另一个目标指标（如 `Water_Quality_Index`）
- **模型**: XGBoost回归（优化版）
- **性能**: R² = 0.97（7个指标平均）
- **位置**: `models/indicator_regression/`

### 3. SHAP 可解释性分析

- **分析维度**: 全局重要性、特征交互、样本解释、决策路径（4个维度）
- **完成模型**: LightGBM分类 + XGBoost回归
- **输出文件**: 57个（2个报告 + 55个PNG图表）
- **位置**: `SHAP_Analysis_20251119_180422/`

## 🗺️ 地图可视化 (Map Visualization)

新增的 `eda_map.py` 脚本提供了一个强大的交互式地理可视化工具，用于探索各省份的水质状况。

- **功能**: 生成一个以省份为单位的多图层交互式地图
- **核心特性**:
  - **分层设色**: 根据不同水质指标（如水质指数、COD、总氮等）的平均值对省份进行颜色渲染
  - **图层切换**: 用户可以在地图控件中方便地切换不同的指标图层，进行对比分析
  - **详细信息**: 点击每个省份的中心标记，可以弹出窗口显示该地区详细的水质统计数据
- **技术栈**: 使用 `folium` 库生成 HTML 地图，并从网络实时获取 GeoJSON 地理数据

## 🚀 快速开始

### 1. 运行探索性数据分析与特征工程

```bash
cd EDA

# 基础EDA
python eda.py
python eda_feature_engineering.py

# 时间序列可视化EDA
python timeseries_visualization_eda.py

# 生成交互式地图
python eda_map.py

# 季节性特征分析
python seasonality_analysis_sarima.py
```

### 2. 运行分类模型（推荐）

```bash
cd models/pollution_level_classification

# 基础模型
python classification_random_forest.py     # 随机森林
python classification_xgboost.py           # XGBoost（最佳）
python classification_catboost.py          # CatBoost
python classification_lightgbm.py          # LightGBM
python classification_svm.py               # SVM
python classification_knn.py               # KNN
python classification_logistic_regression.py # 逻辑回归

# 高级模型（v2.7）
python transformer_classifier.py           # Transformer分类器
python ultimate_ensemble_system.py         # 终极集成系统
```

### 3. 运行回归预测（论文方法）

```bash
cd models/indicator_regression

python regression_xgboost_clean.py         # XGBoost回归（优化版）
python feature_engineering_advanced.py     # 高级特征工程
```

### 4. 生成SHAP可解释性分析

```bash
cd scripts

python generate_shap_analysis.py           # 生成SHAP分析
python generate_paper_figures.py           # 生成论文图表
```

### 5. 运行时间序列模型（实验性，不推荐）

```bash
cd models/timeseries_models_province_revised

# 基础时序模型
python arima_province_revised.py
python sarima_province_revised.py
python prophet_province_revised.py

# 高级时序模型
python lstm_province_revised_auto_gpu.py       # LSTM + GPU加速
python lstm_rf_hybrid_province.py              # LSTM+RF混合
python autoformer_province.py                  # Autoformer（v2.7）

# 批量运行
python run_all.py

# ⚠️ 注意：时间序列方法不适合此数据集，仅用于学术对比
```

## 📊 模型评估总结

### 分类模型（Pollution_Level 预测）

| 模型 | 准确率 (Accuracy) | 核心结论 |
| :--- | :--- | :--- |
| **XGBoost** | **~99.58%** | **性能最佳**。在所有类别上都表现出极高的精确率和召回率。 |
| **随机森林** | **~99.55%** | 性能与XGBoost非常接近，同样表现出色。 |
| **终极集成系统** | **~99.6%+** | 超越单一最佳模型（v2.7新增）。 |
| **Transformer** | **~99%+** | 接近XGBoost水平（v2.7新增）。 |
| **支持向量机 (SVM)** | **~98.16%** | 表现良好，但准确率略低于基于树的集成模型。 |
| **逻辑回归** | **~96.17%** | 作为线性基准模型。 |

### 回归模型（指标预测）

| 方法 | R² (平均) | 核心结论 |
| :--- | :--- | :--- |
| **XGBoost回归（横截面）** | **~0.97** | **性能优异**。适用于本数据集（论文主要方法）。 |
| **LSTM时序预测** | **~-0.1** | 不适合本数据集（采样不规则）。 |
| **LSTM+RF混合** | **~-0.3** | 不适合本数据集（采样不规则）。 |

**总结**: 对于此数据集，**横截面方法（XGBoost、Random Forest）** 是最佳选择。时间序列方法因数据不规则而不适用。

## 🔧 环境配置

### Python依赖

```bash
pip install -r requirements.txt
```

如果缺少特定库，请单独安装：

```bash
pip install xgboost catboost lightgbm shap zhplot
```

### 主要库

- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn, xgboost, catboost, lightgbm
- **深度学习**: torch (PyTorch)
- **可解释性**: shap
- **可视化**: matplotlib, seaborn, folium, zhplot
- **时间序列**: statsmodels, prophet

## 📋 编码规则 & 项目约定

### ⚠️ 重要规则

1. **始终用中文写内容和输出**
   - 🇨🇳 前端界面、变量名注释、API返回信息全部使用中文
   - 🇨🇳 省份显示为中文（北京、广东、河南等），不带"省"字
   - 🇨🇳 与用户的所有交互信息必须是中文
   - ✅ 内部API传参可以用英文（如 province=Beijing），但显示给用户的都是中文

2. **不要创建说明文档**
   - ❌ 不要创建新的 `.md` 说明文件
   - ✅ 所有说明、改动、功能只在本 `CLAUDE.md` 里记录
   - ✅ 代码注释已足够清晰，无需额外文档

3. **模型开发**
   - 优化模型放在对应的模型目录中
   - 使用 Optuna 进行超参数优化
   - 新增模型需要在本文档中更新

4. **⭐⭐⭐ Station级别模型归档规则（v2.5.1）**
   - ❌ **不再开发和维护** Station级别（监测站级别）的时间序列模型
   - ✅ **仅关注 Province级别**（省份级别）的时间序列模型
   - 📦 所有Station模型已归档至 `models/archived/` 目录
   - 🎯 **原因**：
     - Province级别数据更稳定，聚合后噪声更小
     - 180个监测站数据维护成本高，Province级别只需关注10个省份
     - 实际应用中省份级别预测更有实用价值
   - ⚠️ **重要**：任何涉及Station级别的需求，都应引导转向Province级别实现

5. **❌ 代码中不要使用 Emoji**
   - ❌ **禁止在 Python 代码的 print 语句中使用 emoji 符号**（如 📥🔧📍🎯🎉 等）
   - ✅ **使用文本标签替代**：`[OK]`, `[ERROR]`, `[WARN]`, `[INFO]`, `[DONE]`, `[TARGET]`, `[PROVINCE]`
   - 🎯 **原因**：
     - Windows 系统的 GBK 编码不支持大部分 emoji
     - 会导致 `UnicodeEncodeError` 运行时错误
     - 文本标签更稳定、更兼容
   - ⚠️ **重要**：本 CLAUDE.md 文档里可以使用 emoji（Markdown 支持），但代码文件中绝对不能使用

6. **📊 Matplotlib 中文字体配置**
   - ✅ **使用 zhplot 库**（推荐）
   - ✅ **配置方法**：在脚本最开头导入 `zhplot` 即可
   - 🎯 **原因**：
     - zhplot 会自动配置matplotlib的中文字体
     - 无需手动配置字体，一行代码解决
     - 自动处理负号显示问题
     - 比matplotlib内置方法更简单可靠
   - 💻 **标准配置代码**：
     ```python
     # 在import部分的最开头添加
     import zhplot  # 自动配置matplotlib中文字体
     ```
   - ⚠️ **注意**：
     - zhplot 必须在 matplotlib.pyplot 之前导入
     - 无需其他任何字体配置代码
     - 删除所有旧的字体配置代码（plt.rcParams等）

7. **优先使用黑体（SimHei）字体**
   - 原因：黑体在科学出版物中更常用，符合学术规范
   - zhplot 默认使用黑体，无需额外配置

---

**创建日期**: 2025-01-05
**最后更新**: 2025-11-22
**当前版本**: 2.8 (重构版)
**维护者**: Claude Code
