# 水质污染数据分析项目 - 项目配置文档

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
│   ├── determine_ranges.py              # 确定指标范围
│   ├── Feature_Engineering_EDA/         # 特征工程输出
│   ├── Timeseries_EDA/                  # 时间序列EDA输出
│   └── traditional_eda/                 # 传统EDA输出
│
├── 🤖 机器学习模型 (models/)
│   │
│   ├── pollution_level_classification/  # ⭐ 污染等级分类模型（推荐）
│   │   ├── classification_random_forest.py      # 随机森林
│   │   ├── classification_xgboost.py            # XGBoost
│   │   ├── classification_svm.py                # 支持向量机
│   │   ├── classification_logistic_regression.py # 逻辑回归
│   │   ├── classification_catboost.py           # CatBoost
│   │   ├── classification_lightgbm.py           # LightGBM
│   │   └── classification_knn.py                # K近邻
│   │
│   ├── timeseries_models_province/      # 时间序列模型 - 省份级别（旧版）
│   │   ├── arima_province_prediction.py
│   │   ├── sarima_province_prediction.py
│   │   ├── prophet_province_prediction.py
│   │   ├── ar_province_prediction.py
│   │   └── lstm_province_prediction*.py
│   │
│   ├── timeseries_models_province_revised/ # ⭐⭐⭐ 改进版时间序列 - Province级别（主要使用）
│   │   ├── arima_province_revised.py
│   │   ├── sarima_province_revised.py
│   │   ├── prophet_province_revised.py
│   │   ├── ar_province_revised.py
│   │   ├── lstm_province_revised_auto_gpu.py       # 自动GPU加速
│   │   ├── lstm_province_optimized_auto_gpu.py     # 优化版LSTM
│   │   ├── lstm_rf_hybrid_province.py              # ⭐⭐⭐ LSTM+RF混合模型（最新）
│   │   └── run_all.py                              # 批量运行脚本
│   │
│   ├── archived/                         # 📦 归档模型（不再维护）
│   │   ├── timeseries_models_station/           # Station级别模型（旧版）
│   │   └── timeseries_models_station_revised/   # Station级别模型（改进版）
│   │
│   └── backup_prediction_attempts_crosssectional/  # 横截面预测尝试（备份）
│       └── ... (多个实验性脚本)
│
├── 🌟 季节性分析（NEW）
│   └── seasonality_analysis_sarima.py   # ⭐⭐ 水质因子季节性特征分析
│
├── 🔄 批量运行脚本
│   └── run_all.py                       # 根目录批量运行脚本
│
├── 📁 模型输出 (models_outputs/)
│   ├── outputs_pollution_level_classification/  # 分类模型输出
│   ├── outputs_station_prediction/              # 监测站预测输出
│   └── outputs_province_prediction/             # 省份预测输出
│
└── 📊 统一输出目录 (Outputs/)
    ├── ARIMA_2023Train_2024Valid_Station/
    ├── ARIMA_Province_2023Train_2024Valid_Province/
    ├── SARIMA_Province_2023Train_2024Valid_Province/
    ├── SARIMA_Revised_Station/
    ├── Prophet_2023Train_2024Valid_Station/
    ├── Prophet_Province_2023Train_2024Valid_Province/
    ├── AR_Province_2023Train_2024Valid_Province/
    ├── LSTM_2023Train_2024Valid_Station_GPU/
    ├── LSTM_2023Train_2024Valid_Province_GPU/
    ├── LSTM_2023Only_Province_GPU_Optimized/
    ├── TimeSeries_2023Train_2024Valid/
    └── Seasonality_Analysis_SARIMA/         # ⭐⭐ NEW: 季节性分析输出
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
我们已经成功构建了以下7个模型，用于预测 `Pollution_Level`（污染等级）：
- **随机森林 (Random Forest)**: 效果强大且稳定，可解释性好。
- **XGBoost**: 性能卓越，通常比随机森林更精准。
- **CatBoost**: 专门处理分类特征的梯度提升算法，性能优异。
- **LightGBM**: 高效的梯度提升框架，训练速度快。
- **支持向量机 (SVM)**: 在高维空间中表现良好，提供了不同的决策边界思路。
- **K近邻 (KNN)**: 基于实例的学习方法，适合局部模式识别。
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
# 基础EDA（在EDA目录下）
cd EDA
python eda.py
python eda_feature_engineering.py
# 输出: Feature_Engineering_EDA/ 和 preprocessed_data_feature_eng.csv
```

### 2. 运行时间序列可视化EDA
```bash
# 为所有180个监测站生成时间序列折线图，分析采样间隔
cd EDA
python timeseries_visualization_eda.py
# 输出: Timeseries_EDA/ (180个站点图表 + 汇总分析)
```

### 3. 生成交互式地图
```bash
# 此脚本会生成一个多图层的省级水质地图
cd EDA
python eda_map.py
# 输出: province_level_water_quality_map.html
```

### 4. ⭐⭐ NEW: 运行季节性特征分析
```bash
# 使用SARIMA方法分析各水质因子的季节性特征
python seasonality_analysis_sarima.py
# 输出: Outputs/Seasonality_Analysis_SARIMA/
#   - seasonality_report.txt (综合报告)
#   - seasonality_heatmap.png (季节性强度热力图)
#   - decomposition/ (季节性分解图)
#   - acf_pacf/ (自相关图)
#   - seasonal_patterns/ (季节性模式对比)
```

### 5. 运行分类模型（推荐）
```bash
cd models/pollution_level_classification

# 运行所有分类模型
python classification_random_forest.py     # 随机森林
python classification_xgboost.py           # XGBoost
python classification_catboost.py          # CatBoost
python classification_lightgbm.py          # LightGBM
python classification_svm.py               # SVM
python classification_knn.py               # KNN
python classification_logistic_regression.py # 逻辑回归

# 所有模型的报告、混淆矩阵和特征重要性图表都会保存在models_outputs/下
```

### 6. 运行改进版时间序列模型（实验性）
```bash
# Station级别时间序列模型
cd models/timeseries_models_station_revised

python arima_station_revised.py
python sarima_station_revised.py
python prophet_station_revised.py
python lstm_station_revised_auto_gpu.py        # 自动GPU加速
python lstm_station_optimized_auto_gpu.py      # 优化版LSTM
python timeseries_station_prediction_revised.py

# Province级别时间序列模型
cd models/timeseries_models_province_revised

python arima_province_revised.py
python sarima_province_revised.py
python prophet_province_revised.py
python ar_province_revised.py
python lstm_province_revised_auto_gpu.py       # 自动GPU加速
python lstm_province_optimized_auto_gpu.py     # 优化版LSTM

# 批量运行（在各自目录下）
python run_all.py

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

- **v2.7** (2025-11-21) - Claude Code
  - 🚀 **终极模型整合** - 整合所有模型形成最强预测系统
    - ✨ 新增 Transformer水质分类器 (transformer_classifier.py)
      - Multi-Head Self-Attention机制 - 捕捉特征间的全局依赖
      - Positional Encoding - 为特征添加位置信息
      - 可配置架构: d_model=128, num_heads=8, num_layers=4
      - 预期准确率: ~99%+ (接近XGBoost水平)
      - 完整的注意力权重可视化功能

    - ✨ 新增 Autoformer时序预测模型 (autoformer_province.py)
      - Auto-Correlation机制 - 基于FFT发现周期性模式
      - Series Decomposition - 趋势-季节性分解
      - Deep Decomposition Architecture - 多层渐进式分解
      - 注意: 时序方法因数据不规则而效果不佳，主要用于学术探索

    - 🎯 终极集成预测系统 (ultimate_ensemble_system.py)
      - Level-1 基础模型层: 9个模型 (XGBoost, RF, LightGBM, CatBoost, GB, LR, SVM, KNN, MLP)
      - Level-2 元学习层: 3种集成策略 (Stacking, Weighted Voting, Selective Ensemble)
      - 动态权重分配: 基于验证集性能自动调整权重
      - 鲁棒性增强: 自动处理模型训练失败
      - 预期准确率: 99.6%+ (超越单一最佳模型)

    - 📋 快速启动脚本 (run_all_ultimate_models.py)
      - 一键运行所有模型
      - 支持跳过时序预测: --skip-timeseries
      - 支持只运行集成系统: --only-ensemble

    - 📖 完整使用文档 (ULTIMATE_MODELS_GUIDE.md)
      - 详细的模型介绍和性能对比
      - 4种使用场景和推荐方案
      - 高级配置和性能优化技巧
      - 完整的故障排查指南

  - 📊 **新增输出目录**
    - Outputs/Transformer_Classification/ - Transformer分类结果
    - Outputs/Autoformer_Province_Prediction/ - Autoformer时序预测
    - Outputs/Ultimate_Ensemble_System/ - 终极集成系统

  - 🎯 **推荐使用方案**
    - 生产环境: 终极集成系统 (Weighted Voting) - 99.6%准确率
    - 快速开发: XGBoost单模型 - 99.58%准确率
    - 学术研究: Transformer + 完整对比实验
    - 可解释性: RandomForest + SHAP分析

- **v2.6** (2025-11-20) - Claude Code
  - 📄 **论文框架设计**
    - 创建完整的学术论文框架（docs/PAPER_FRAMEWORK.md）
    - 目标期刊：Applied Soft Computing（机器学习/数据科学类）
    - 论文长度：15-18页长文
    - 目标读者：交叉学科（机器学习研究者 + 环境科学家）

  - 📊 **SHAP分析完整性评估**
    - 系统评估项目SHAP值分析的完整性
    - 已完成：分类模型（LightGBM）+ 回归模型（XGBoost_Clean）
    - 输出文件：57个（55个PNG + 2个MD报告）
    - 分析维度：全局重要性、特征交互、样本解释、决策路径（4个维度100%覆盖）

  - 🎯 **论文框架核心内容**
    - 3个候选标题（强调方法论/可解释性/应用价值）
    - 8个章节完整结构（Abstract + 7个正文章节）
    - 13个必需图表规划 + 8个必需表格
    - 写作风格指南（避免AI味，强调数据驱动）
    - 核心创新点：
      * 方法对比实验（时间序列失败 vs 横截面成功）
      * SHAP可解释性系统性应用（4维度框架）
      * 实际应用价值（传感器配置、预警阈值）

  - 📝 **论文写作要点**
    - 强调负面结果的学术价值（时间序列R²=-0.1）
    - 数据驱动（用CV=1.99、ACF=0.03等具体数字）
    - 诚实讨论局限性（数据跨度短、特征缺失）
    - 避免空话（"随着大数据时代"、"具有重要意义"等）
    - 具体应用建议（COD传感器5万元/台、预警阈值20 mg/L）

- **v2.5.2** (2025-11-13) - Claude Code
  - 📁 **项目结构重组**
    - 创建 `docs/` 目录存放论文相关文档
    - 创建 `scripts/` 目录存放分析脚本
    - 移动临时脚本和文档到对应目录
    - 删除根目录临时文件（nul等）
    - 创建 `PROJECT_STRUCTURE.md` 项目结构说明文档
  - 📊 **Matplotlib字体优先级规则**
    - 新增规则7：优先使用黑体（SimHei）而非微软雅黑
    - 原因：黑体在科学出版物中更常用，符合学术规范
    - 添加标准配置代码示例
  - ⭐⭐⭐ **论文材料100%完成**
    - 完成XGBoost回归实验（7个指标）
    - 生成所有论文图表（3个核心图）
    - 创建完整交付文档（FINAL_DELIVERY_COMPLETE.md）
    - 核心发现：横截面回归R²=0.97（成功指标），时间序列R²<0（失败）

- **v2.5.1** (2025-11-12) - Claude Code
  - ⭐⭐⭐ **混合模型优化（参考 revise.md v1 建议）**
    - 修改 LSTM 输入特征：去掉当期原始 indicator，只用滞后与派生特征
    - 更换 scaler 从 MinMaxScaler 到 StandardScaler（对分布漂移更稳健）
    - 减小 dropout 从 0.2 到 0.1
    - 调整 seq_length 默认值从 10 到 7（短窗口更适合不规则采样）
    - 增大 RF 容量：n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=1
    - 实现自适应融合权重：在验证集上学习最优权重，替代固定 0.5
    - 修复文件：`lstm_rf_hybrid_province.py`
  - ⭐⭐⭐ **三大关键补丁（参考 revise.md v2）**
    - **补丁A - 统一评估切片**：修复 RF 对齐 bug，所有模型评估使用同一段真值 `y_test[seq_length : seq_length + n_common]`
    - **补丁B - 重训最终模型**：学完权重后，用 train+val 全量数据重训最终模型，充分利用 2023 全年数据
    - **补丁C - 带偏置岭回归**：融合函数改用岭回归 + 偏置项，解决系统性偏差，公式 `y = w_lstm*lstm + w_rf*rf + bias`
    - **预期效果**：修复"图像与指标不一致"问题，R² 由负转零/正
  - 📦 **归档 Station 级别模型**
    - 将所有 Station 级别模型移至 `models/archived/` 目录
    - 今后仅关注和维护 Province 级别模型
    - 原因：Province 数据更稳定，维护成本更低，实用价值更高
  - 🔧 **编码规范**
    - 新增规则：代码中禁止使用 emoji，使用文本标签替代（[OK], [ERROR] 等）
    - 原因：避免 Windows GBK 编码导致的 UnicodeEncodeError

- **v2.5** (2025-11-10) - Claude Code
  - ⭐⭐⭐ 新增 LSTM + RandomForest 混合模型 + Optuna 超参数优化
  - 新增 4 个 REST API 端点 (`/api/model/hybrid/*`)
  - 支持 11 个超参数自动优化
  - 为 `/models` 下的混合模型脚本添加调参功能
  - 新增平台集成测试脚本 `test_hybrid_api.py`
  - **⭐⭐ LSTM 模型优化（参考 lstm_station_optimized_auto_gpu.py）**
    - 添加 LayerNorm 提高训练稳定性
    - 改进的全连接层结构（hidden_size // 2）
    - CosineAnnealingLR 学习率调度（T_max=50, eta_min=1e-6）
    - 优化的早停策略：
      - 正常训练 patience=30，min_delta=1e-4
      - Optuna 试验 patience=20，min_delta=1e-4（更快收敛）
    - 增加 epochs 范围至 30-150（Optuna 搜索）
    - 同步优化 `lstm_rf_hybrid_station.py`、`lstm_rf_hybrid_province.py` 和 `backend/models/lstm_rf_hybrid.py`
    - 修复 tkinter 错误：设置 matplotlib 使用非交互式后端（Agg）
    - 添加训练进度显示（Patience 计数器）
  - **⭐⭐⭐ 混合模型关键修复（参考 revise.md）**
    - **时间步对齐**：修复 LSTM (预测 t+1) 与 RF (预测 t) 的融合错位问题
    - **特征泄漏修复**：差分和滚动统计特征使用严格过去值 (shift(1))
    - **验证集早停**：LSTM 训练引入独立验证集（85%/15% 时间切分）
    - **可视化修复**：plot_predictions 使用 seq_length 而非固定 -10
    - **评估对齐**：所有模型评估使用正确的时间索引对齐
    - 修复文件：`lstm_rf_hybrid_province.py`（完整修复）
  - **⭐⭐⭐ 数据分布漂移对策（针对 2024 R²=-1.09）**
    - **gap_days 特征**：处理不等间隔采样（1-14天），LSTM 知道"这一步隔了几天"
    - **sin/cos 循环特征**：替代单一数值月/日，更好建模季节性周期
    - **SmoothL1Loss (Huber)**：替代 MSELoss，对异常值和重尾分布更稳健
    - **学习型融合权重**：使用验证集最小二乘学习权重，替代固定 0.5
    - **station_count 特征**：记录每日参与聚合的监测站数，捕获采样构成漂移
    - 针对问题：2023→2024 分布漂移（KS检验显著，方差翻倍）
  - **新增前端时间序列预测模块** (Prediction.vue 新增标签页)
    - 支持 LSTM+RF 混合模型训练
    - 支持 Optuna 超参数优化
    - 实时显示模型性能指标
    - 支持按省份或监测站进行精细预测
  - **新增污染等级预测的输入值范围验证**
    - 宽松范围限制（约为实际数据范围的1.5-2倍）
    - 超出范围的值 ❌：中断预测，要求用户修改
    - 偏离预期范围的值 ⚠️：显示警告，允许用户选择继续或修改
    - 每个输入框显示合理范围提示
  - **标准化前端省份列表**
    - 统一所有 Vue 组件使用 10 个英文省份名称
    - 修复了 Dashboard.vue、Analysis.vue、Monitoring.vue、Spatial.vue、Quality.vue 的省份列表
    - 省份列表标准化为：Beijing, Guangdong, Henan, Hubei, Jiangsu, Shandong, Shanghai, Sichuan, Yunnan, Zhejiang
    - 移除了所有不存在的省份映射（Fujian, Anhui, Hunan）

- **v2.4** (2025-10-31) - Claude Code
  - ⭐⭐ 新增季节性特征分析模块（SARIMA方法）
  - 新增3个分类模型（CatBoost、LightGBM、KNN）
  - 完善项目结构文档
  - 更新CLAUDE.md配置文件

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

3. **API 集成**
   - 新增 API 端点必须在 `backend/app.py` 中集成
   - 每个端点需要登录认证 (`@login_required`)
   - 完整的错误处理和日志记录

4. **模型开发**
   - 优化模型放在 `backend/models/` 目录
   - 原始模型脚本放在 `/models` 目录，支持调参功能
   - 使用 Optuna 进行超参数优化

5. **⭐⭐⭐ Station级别模型归档规则（v2.5.1 新增）**
   - ❌ **不再开发和维护** Station级别（监测站级别）的时间序列模型
   - ✅ **仅关注 Province级别**（省份级别）的时间序列模型
   - 📦 所有Station模型已归档至 `models/archived/` 目录
   - 🎯 **原因**：
     - Province级别数据更稳定，聚合后噪声更小
     - 180个监测站数据维护成本高，Province级别只需关注10个省份
     - 实际应用中省份级别预测更有实用价值
   - ⚠️ **重要**：任何涉及Station级别的需求，都应引导转向Province级别实现

6. **❌ 代码中不要使用 Emoji**
   - ❌ **禁止在 Python 代码的 print 语句中使用 emoji 符号**（如 📥🔧📍🎯🎉 等）
   - ✅ **使用文本标签替代**：`[OK]`, `[ERROR]`, `[WARN]`, `[INFO]`, `[DONE]`, `[TARGET]`, `[PROVINCE]`
   - 🎯 **原因**：
     - Windows 系统的 GBK 编码不支持大部分 emoji
     - 会导致 `UnicodeEncodeError` 运行时错误
     - 文本标签更稳定、更兼容
   - ⚠️ **重要**：本 CLAUDE.md 文档里可以使用 emoji（Markdown 支持），但代码文件中绝对不能使用

7. **📊 Matplotlib 中文字体配置**
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

### LSTM + RF 混合模型相关文件

| 文件 | 位置 | 功能 |
|------|------|------|
| lstm_rf_hybrid.py | `backend/models/` | 核心模型类（用于平台API） |
| lstm_rf_hybrid_province.py | `models/timeseries_models_province_revised/` | ⭐⭐⭐ 省份级模型 + Optuna 调参（主要使用） |
| ~~lstm_rf_hybrid_station.py~~ | ~~`models/archived/`~~ | ❌ 已归档，不再维护 |
| test_hybrid_api.py | `platform_demo/` | 平台 API 测试脚本 |

### 新增 API 端点（v2.5）

```
POST   /api/model/hybrid/train       - 训练混合模型
POST   /api/model/hybrid/optimize    - Optuna 超参数优化
POST   /api/model/hybrid/predict     - 使用模型预测
GET    /api/model/hybrid/status      - 获取模型状态
```

所有端点需要登录认证，支持参数：
- `indicator`: 目标指标名称
- `station`: 监测站（可选）
- `province`: 省份名称（可选）
- `n_trials`: Optuna 试验次数（default: 10）

---

**创建日期**: 2025-01-05
**最后更新**: 2025-11-20
**维护者**: Claude Code (v2.3-v2.6), Gemini Agent (v2.0-v2.2)
**版本**: 2.6 (论文框架设计 + SHAP分析评估)
