# 中国水质监测数据集分析项目 | China Water Quality Dataset Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Status](https://img.shields.io/badge/Status-Archived-lightgrey)

> **本项目已结束并归档。**
> 所有开发活动已停止，代码库仅作展示与参考用途。

一个基于机器学习与数据分析的水质污染监测与预测系统，包含完整的探索性数据分析（EDA）、多种机器学习模型、可解释性分析（SHAP）以及交互式可视化平台。

---

## 📋 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [数据集说明](#数据集说明)
- [模型性能](#模型性能)
- [可视化平台](#可视化平台)
- [文档与报告](#文档与报告)
- [版本历史](#版本历史)
- [贡献指南](#贡献指南)

---

## 🎯 项目概述

本项目针对中国水污染监测数据集进行深度分析，旨在：

1. **探索水质分布规律** - 时空分布、区域差异、季节性特征
2. **预测污染等级** - 基于水质指标预测污染等级（优/良/中等/差/极差）
3. **预测水质指标** - 时间序列预测未来水质变化趋势
4. **可解释性分析** - 通过SHAP值解释模型决策过程
5. **可视化展示** - 交互式Web平台实时展示监测数据

### 数据规模

- **数据量**: 15,524条水质监测记录
- **覆盖范围**: 10个省份，180+个监测站点
- **时间跨度**: 2023-2024年
- **水质指标**: 16个原始指标 + 3个工程特征

### 核心成果

- ✅ **污染等级分类准确率**: 99.58% (XGBoost)
- ✅ **化学污染物回归R²**: 0.97 (XGBoost)
- ✅ **SHAP可解释性分析**: 4维度完整分析框架
- ✅ **交互式Web平台**: Vue3 + Flask全栈应用

---

## 🚀 核心功能

### 1. 探索性数据分析 (EDA)

**位置**: `EDA/` 目录

- **基础统计分析** (`eda.py`)
  - 数据分布、缺失值分析
  - 水质指标描述性统计
  - 区域与时间维度基础分析

- **时空分布分析**
  - `spatial_distribution_analysis.py` - 空间聚类与地理分布
  - `temporal_distribution_analysis.py` - 时间序列特征提取
  - `eda_map.py` - 交互式地图可视化（Folium）

- **专题分析**
  - `bayesian_network_analysis.py` - 贝叶斯网络因果关系分析
  - `seasonality_analysis_sarima.py` - SARIMA季节性分解
  - `timeseries_visualization_eda.py` - 时间序列可视化
  - `generate_acf_plot.py` - 自相关性分析

### 2. 机器学习模型

**位置**: `models/` 目录

#### 2.1 污染等级分类 (`pollution_level_classification/`)

| 模型 | 准确率 | 特点 |
|:-----|:-------|:-----|
| **XGBoost** | 99.58% | 最佳性能，生产推荐 |
| **LightGBM** | 99.52% | 高速训练，可解释性强 |
| **RandomForest** | 99.45% | 稳定性好，特征重要性直观 |
| **CatBoost** | 99.40% | 自动处理类别特征 |
| **Transformer** | 99%+ | Multi-Head Attention，学术探索 |
| **MLP/SVM/KNN** | 95%+ | 基准对比模型 |

**终极集成系统** (`ultimate_ensemble_system.py`):
- 9个基础模型 + 3种集成策略
- Weighted Voting准确率: **99.6%+**

#### 2.2 水质指标回归 (`indicator_regression/`)

- **XGBoost回归** - 化学污染物R²=0.97
- **RandomForest回归** - 稳健性强
- 支持7个目标指标预测（COD、总氮、氨氮等）

#### 2.3 时间序列预测 (`timeseries_models_province_revised/`)

- **LSTM + RandomForest混合模型**
  - 自适应融合权重
  - Optuna超参数优化
  - GPU加速支持（CUDA/MPS）
  
- **Autoformer** - Auto-Correlation + Series Decomposition
- **Province级别聚合** - 解决station级别数据不规则问题

> ⚠️ **注意**: 由于数据不规则采样（时间间隔1-14天）和短时跨度（2年），时间序列模型效果有限（R²≈-0.1），主要用于学术探索。横截面方法性能更优。

### 3. 可解释性分析

**位置**: `scripts/generate_shap_analysis.py`

**完整报告**: [docs/SHAP_INTERPRETABILITY_ANALYSIS.md](./docs/SHAP_INTERPRETABILITY_ANALYSIS.md)

#### 分析维度

1. **全局特征重要性** - Top 5关键因素识别
2. **特征交互分析** - COD×浊度、总氮×NP比等
3. **样本级别解释** - 36+3个典型案例
4. **决策路径可视化** - TreeExplainer决策树路径

#### 核心发现

- **化学需氧量（COD）** - 最重要预测因子
- **浊度×COD交互** - 强协同效应
- **总氮/总磷比** - 富营养化关键指标
- **业务建议** - 传感器配置优先级、预警阈值设定

### 4. 交互式可视化平台

**位置**: `platform_demo/`

**完整文档**: [platform_demo/README.md](./platform_demo/README.md)

#### 技术架构

- **前端**: Vue 3 + Element Plus + Leaflet.js + Chart.js
- **后端**: Flask 2.3.0 + RESTful API
- **数据库**: SQLite（用户管理）

#### 主要功能

1. **动态地图展示** 🗺️
   - 180+监测站点实时渲染
   - 污染等级颜色编码
   - 站点聚类优化

2. **时间轴控制** ⏰
   - 日/周/月多粒度查询
   - 趋势图表动态更新

3. **空间查询** 📍
   - 矩形/圆形区域选择
   - 实时统计计算

4. **预测服务** 🔮
   - 污染等级预测（XGBoost模型）
   - 水质指标时序预测（LSTM+RF混合）
   - Optuna超参数在线优化

#### 快速启动

```bash
cd platform_demo
python start_all.py
# 访问 http://localhost:5000
```

---

## 📁 项目结构

```
china-water-quality-dataset-analysis/
├── 中国水污染监测数据集.csv          # 原始数据集
├── preprocessed_data.csv              # 预处理后数据
├── preprocessed_data_feature_eng.csv  # 特征工程后数据
├── requirements.txt                   # Python依赖
├── CHANGELOG.md                       # 版本历史
│
├── preprocess/                        # 数据预处理
│   ├── preprocess.py                  # 预处理脚本
│   └── .env.example                   # API密钥示例
│
├── EDA/                               # 探索性数据分析
│   ├── eda.py                         # 基础EDA
│   ├── spatial_distribution_analysis.py
│   ├── temporal_distribution_analysis.py
│   ├── bayesian_network_analysis.py
│   ├── seasonality_analysis_sarima.py
│   └── eda_map.py                     # 地图可视化
│
├── models/                            # 机器学习模型
│   ├── pollution_level_classification/  # 分类模型
│   │   ├── xgboost_classifier.py
│   │   ├── lightgbm_classifier.py
│   │   ├── rf_classifier.py
│   │   ├── transformer_classifier.py
│   │   └── ultimate_ensemble_system.py
│   ├── indicator_regression/          # 回归模型
│   │   └── xgboost_regressor.py
│   ├── timeseries_models_province_revised/  # 时序模型
│   │   ├── lstm_province_optimized_auto_gpu.py
│   │   ├── lstm_rf_hybrid_province.py
│   │   └── autoformer_province.py
│   └── archived/                      # 已归档模型
│
├── scripts/                           # 分析脚本
│   ├── generate_shap_analysis.py      # SHAP分析
│   ├── auto_select_best_interpretable_model.py
│   ├── verify_dataset_authenticity.py  # 数据真实性验证
│   ├── generate_correlation_heatmap.py
│   ├── generate_paper_figures.py      # 论文图表生成
│   └── run_all.py                     # 批量运行
│
├── platform_demo/                     # 可视化平台
│   ├── backend/                       # Flask后端
│   │   ├── app.py
│   │   ├── models/
│   │   └── data/
│   ├── frontend-vue/                  # Vue前端
│   │   ├── src/
│   │   └── public/
│   ├── start_all.py                   # 一键启动
│   └── test_hybrid_api.py             # API测试
│
├── docs/                              # 项目文档
│   ├── README.md                      # 文档索引
│   ├── SHAP_INTERPRETABILITY_ANALYSIS.md  # ⭐ SHAP完整报告
│   ├── MODEL_VERIFICATION_REPORT_20251128.md
│   ├── API_REFERENCE.md               # API文档
│   ├── analysis_reports/              # 分析报告
│   ├── model_experiments/             # 实验记录
│   ├── project_management/            # 项目管理
│   └── data_documentation/            # 数据说明
│
├── Outputs/                           # 模型输出
│   ├── EDA/                           # EDA图表
│   ├── Classification_XGBoost/        # 分类模型结果
│   ├── Regression_XGBoost/            # 回归模型结果
│   ├── SHAP_Analysis/                 # SHAP分析结果
│   ├── Transformer_Classification/
│   ├── Autoformer_Province_Prediction/
│   └── Ultimate_Ensemble_System/
│
└── todo/                              # 待办事项
```

---

## 💻 技术栈

### 数据科学

- **数据处理**: Pandas 2.3.3, NumPy 2.3.3
- **可视化**: Matplotlib 3.10.6, Seaborn 0.13.2, Folium
- **地理空间**: GeoPandas 1.1.1, Shapely 2.1.2

### 机器学习

- **经典模型**: Scikit-learn 1.7.2
- **梯度提升**: XGBoost, LightGBM, CatBoost
- **深度学习**: TensorFlow 2.20.0, PyTorch 2.8.0
- **可解释性**: SHAP
- **超参数优化**: Optuna

### Web开发

- **后端**: Flask 2.3.0, Flask-CORS
- **前端**: Vue 3, Element Plus
- **地图**: Leaflet.js
- **图表**: Chart.js

---

## 🚀 快速开始

### 1. 环境配置

**要求**: Python 3.8+

```bash
# 克隆项目
git clone <repository-url>
cd china-water-quality-dataset-analysis

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
cd preprocess
python preprocess.py
```

生成文件:
- `preprocessed_data.csv` - 清洗后数据
- `preprocessed_data_feature_eng.csv` - 特征工程数据

### 3. 运行EDA分析

```bash
cd EDA
python eda.py  # 基础EDA
python eda_map.py  # 地图可视化
python bayesian_network_analysis.py  # 贝叶斯网络
```

### 4. 训练模型

**污染等级分类 (推荐):**
```bash
cd models/pollution_level_classification
python xgboost_classifier.py  # 单模型 (99.58%)
python ultimate_ensemble_system.py  # 集成系统 (99.6%+)
```

**水质指标回归:**
```bash
cd models/indicator_regression
python xgboost_regressor.py
```

**时间序列预测:**
```bash
cd models/timeseries_models_province_revised
python lstm_rf_hybrid_province.py --province Beijing
```

### 5. 生成SHAP分析

```bash
cd scripts
python generate_shap_analysis.py
```

输出: `Outputs/SHAP_Analysis/` + `docs/SHAP_INTERPRETABILITY_ANALYSIS.md`

### 6. 启动可视化平台

```bash
cd platform_demo
python start_all.py
# 访问 http://localhost:5000
```

---

## 📊 数据集说明

### 数据来源

中国水污染监测数据集（2023-2024年），包含全国10个省份180+个监测站点的水质监测数据。

### 16个水质指标

#### 化学污染物
- **COD (mg/L)** - 化学需氧量
- **BOD (mg/L)** - 生化需氧量
- **氨氮 (mg/L)** - NH₃-N
- **总氮 (mg/L)** - TN
- **总磷 (mg/L)** - TP

#### 重金属
- **铅 (mg/L)** - Pb
- **汞 (mg/L)** - Hg
- **六价铬 (mg/L)** - Cr⁶⁺

#### 物理参数
- **浊度 (NTU)**
- **溶解氧 (mg/L)** - DO
- **电导率 (μS/cm)**

#### 其他
- **pH值**
- **水质指数 (WQI)** - 综合指标
- **粪大肠菌群 (CFU/100mL)**
- **硝酸盐氮 (mg/L)**
- **氰化物 (mg/L)**

### 工程特征（特征工程后）

- **N/P比** - 总氮/总磷
- **DO饱和度** - 溶解氧相对饱和度
- **有机污染指数** - COD/BOD

### 污染等级

| 等级 | 英文 | WQI范围 | 描述 |
|:-----|:-----|:--------|:-----|
| 优 | Excellent | 0-25 | 水质优良，适合直接饮用 |
| 良 | Good | 25-50 | 水质良好，轻度处理可饮用 |
| 中等 | Fair | 50-75 | 水质一般，需处理后使用 |
| 差 | Poor | 75-100 | 水质较差，污染明显 |
| 极差 | Very Poor | >100 | 水质极差，严重污染 |

---

## 📈 模型性能

### 污染等级分类

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 训练时间 |
|:-----|:-------|:-------|:-------|:-------|:---------|
| **XGBoost** | **99.58%** | 99.59% | 99.58% | 99.58% | ~15s |
| LightGBM | 99.52% | 99.53% | 99.52% | 99.52% | ~8s |
| RandomForest | 99.45% | 99.46% | 99.45% | 99.45% | ~20s |
| CatBoost | 99.40% | 99.41% | 99.40% | 99.40% | ~30s |
| Transformer | 99%+ | - | - | - | ~2min |
| **Ultimate Ensemble** | **99.6%+** | 99.6%+ | 99.6%+ | 99.6%+ | ~45s |

### 水质指标回归

| 目标指标 | R² | RMSE | MAE | 特点 |
|:---------|:---|:-----|:----|:-----|
| **COD** | **0.97** | 2.34 | 1.52 | 最佳预测性能 |
| 总氮 | 0.96 | 0.78 | 0.45 | 高准确性 |
| 氨氮 | 0.95 | 0.23 | 0.15 | 稳定预测 |
| 总磷 | 0.94 | 0.12 | 0.08 | 良好性能 |
| pH | 0.82 | 0.35 | 0.22 | 中等性能 |
| 溶解氧 | -0.03 | 2.45 | 1.89 | 预测困难* |
| 浊度 | -0.15 | 15.2 | 10.3 | 预测困难* |

> \* 生物/物理指标受多因素影响，横截面方法预测困难

### 时间序列预测

| 方法 | R² (2024验证集) | 优点 | 缺点 |
|:-----|:----------------|:-----|:-----|
| LSTM+RF混合 | -0.1 ~ 0.05 | GPU加速，自动调参 | 数据不规则 |
| Autoformer | -0.2 ~ 0.0 | Auto-Correlation | 时间跨度短 |

> ⚠️ 由于数据采样不规则（1-14天间隔）且时间跨度短（2年），时间序列方法效果有限。**推荐使用横截面分类/回归方法。**

---

## 🌐 可视化平台

### 界面预览

平台包含以下模块:

1. **Dashboard** - 数据概览与关键指标
2. **地图监测** - 交互式站点地图
3. **时间分析** - 趋势图表与预测
4. **空间分析** - 区域查询与聚类
5. **污染预测** - 等级预测与指标预测
6. **数据管理** - 上传/导出数据

### API端点

**基础URL**: `http://localhost:5000/api`

| 端点 | 方法 | 功能 |
|:-----|:-----|:-----|
| `/stations` | GET | 获取所有监测站 |
| `/data/timeline` | GET | 时间轴数据 |
| `/data/spatial` | POST | 空间查询 |
| `/predict/pollution_level` | POST | 污染等级预测 |
| `/predict/water_quality` | POST | 水质指标预测 |
| `/model/hybrid/train` | POST | 训练混合模型 |
| `/model/hybrid/optimize` | POST | Optuna优化 |
| `/statistics/summary` | GET | 统计摘要 |
| `/trends` | GET | 趋势数据 |

**完整文档**: [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)

---

## 📚 文档与报告

### 核心文档

| 文档 | 描述 | 位置 |
|:-----|:-----|:-----|
| **SHAP可解释性分析** | 4维度完整分析，58个图表 | [docs/SHAP_INTERPRETABILITY_ANALYSIS.md](./docs/SHAP_INTERPRETABILITY_ANALYSIS.md) |
| 模型验证报告 | 所有模型性能对比 | [docs/MODEL_VERIFICATION_REPORT_20251128.md](./docs/MODEL_VERIFICATION_REPORT_20251128.md) |
| API参考 | RESTful API完整文档 | [docs/API_REFERENCE.md](./docs/API_REFERENCE.md) |
| 平台使用手册 | Web平台功能介绍 | [platform_demo/README.md](./platform_demo/README.md) |
| 项目结构说明 | 目录组织与文件说明 | [docs/project_management/PROJECT_STRUCTURE.md](./docs/project_management/PROJECT_STRUCTURE.md) |

### 分析报告

- **data_leakage_final_report** - 数据泄露检测与修复
- **improvement_summary** - 模型改进总结
- **paper_experiments** - 学术实验记录

### 实验记录

- **revise.md** - 混合模型修订建议
- **data_leakage_checklist** - 泄露检测清单

**完整索引**: [docs/README.md](./docs/README.md)

---

## 📝 版本历史

### v3.0 Final (2026-01-10) - 项目归档

- 🏁 项目正式结束并归档
- 📄 完善最终文档与README
- 🔒 锁定所有模型与数据集状态

### v2.7 (2025-11-21) - 终极模型整合

- ✨ 新增Transformer分类器（Multi-Head Attention）
- ✨ 新增Autoformer时序预测（Auto-Correlation）
- 🎯 终极集成系统（9模型+3策略，准确率99.6%+）
- 📋 快速启动脚本与完整使用文档

### v2.6 (2025-11-20) - 论文框架设计

- 📄 完整学术论文框架（Applied Soft Computing期刊）
- 📊 SHAP分析完整性评估（4维度100%覆盖）
- 📈 13个必需图表 + 8个必需表格规划

### v2.5.2 (2025-11-13) - 项目结构重组

- 📁 创建`docs/`和`scripts/`目录
- 📄 项目结构说明文档（PROJECT_STRUCTURE.md）
- ⭐⭐⭐ XGBoost回归实验完成（7个指标）

### v2.5.1 (2025-11-12) - 混合模型优化

- 修改LSTM输入特征，去除当期原始indicator
- StandardScaler替代MinMaxScaler
- 三大关键补丁（统一评估切片、重训最终模型、带偏置岭回归）
- 📦 归档Station级别模型

### v2.5 (2025-11-10) - LSTM+RF混合模型

- ⭐⭐⭐ LSTM + RandomForest混合模型
- Optuna超参数优化（11个参数）
- 新增4个REST API端点（`/api/model/hybrid/*`）
- LayerNorm + CosineAnnealingLR优化

### v2.4 (2025-10-31) - 季节性分析

- ⭐⭐ SARIMA季节性特征分析
- 新增3个分类模型（CatBoost, LightGBM, KNN）

### v2.3 (2025-10-25) - 时间序列可视化

- ⭐⭐ 时间序列可视化EDA
- 改进版时序模型（2023训练/2024验证）
- LSTM自动GPU加速支持（CUDA/MPS）

### v2.2 (2025-10-15) - 地图可视化

- Folium交互式地图
- 站点聚类显示

### v2.0 (2025-01-05) - 初始版本

- 项目初始化
- 基础EDA与预处理
- 基础分类模型

**完整历史**: [CHANGELOG.md](./CHANGELOG.md)

---

## 🎓 学术应用

### 适用场景

- ✅ **机器学习论文** - 模型对比、可解释性分析
- ✅ **数据科学竞赛** - 完整的特征工程与模型pipeline
- ✅ **环境工程研究** - 水质预测与污染因素分析
- ✅ **课程项目** - 数据分析、机器学习课程设计

### 核心创新点

1. **方法对比实验** - 时间序列失败 vs 横截面成功（有系统性证据）
2. **SHAP可解释性框架** - 4维度系统性应用（全局→交互→局部→路径）
3. **混合模型架构** - LSTM+RF自适应融合
4. **实际应用价值** - 传感器配置建议、预警阈值设定

### 相关资源

- SHAP官方文档: https://shap.readthedocs.io/
- LightGBM文档: https://lightgbm.readthedocs.io/
- XGBoost文档: https://xgboost.readthedocs.io/
- Optuna文档: https://optuna.readthedocs.io/

---

## 🤝 贡献指南

本项目已归档，不再接受新的PR。

---

## 🙏 致谢

- **数据处理**: Pandas, NumPy, GeoPandas
- **机器学习**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **深度学习**: TensorFlow, PyTorch
- **可视化**: Matplotlib, Seaborn, Folium, Leaflet.js, Chart.js
- **Web框架**: Flask, Vue.js, Element Plus
- **可解释性**: SHAP
- **优化**: Optuna

---

**⭐ 感谢大家的支持！**
