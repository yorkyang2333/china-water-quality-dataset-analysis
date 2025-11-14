# 项目结构说明

**Project Structure Documentation**

---

**最后更新**: 2025-11-13
**版本**: v2.5.2

---

## 📁 根目录结构

```
china-water-quality-dataset-analysis/
│
├── 📂 数据文件
│   ├── 中国水污染监测数据集.csv              # 原始数据集
│   ├── 中国水污染监测数据集说明.docx          # 数据集说明文档
│   ├── preprocessed_data.csv                # 预处理后的数据
│   └── preprocessed_data_feature_eng.csv    # 特征工程后的数据
│
├── 📂 核心目录
│   ├── .claude/                  # Claude配置文件
│   │   ├── CLAUDE.md             # 项目配置和编码规范
│   │   └── todos/                # 任务列表
│   │
│   ├── .git/                     # Git版本控制
│   ├── .gitignore                # Git忽略规则
│   │
│   ├── docs/                     # ⭐ 论文和文档目录
│   │   ├── PAPER_EXPERIMENTS.md          # 论文实验报告 (30KB)
│   │   ├── PAPER_DELIVERY_SUMMARY.md     # 论文材料交付总结
│   │   ├── FINAL_DELIVERY_COMPLETE.md    # 最终交付完整总结
│   │   └── revise.md                     # 修订历史文档
│   │
│   ├── scripts/                  # ⭐ 分析脚本目录
│   │   ├── analyze_failure.py            # 失败分析脚本
│   │   ├── deep_analysis.py              # 深度分析脚本 v1
│   │   ├── deep_analysis_v2.py           # 深度分析脚本 v2
│   │   ├── generate_paper_figures.py     # 论文图表生成脚本
│   │   ├── run_all.py                    # 批量运行脚本
│   │   ├── run_lstm_rf_hybrid.py         # LSTM+RF混合模型运行脚本
│   │   └── seasonality_analysis_sarima.py # 季节性分析脚本
│   │
│   ├── preprocess/               # 数据预处理模块
│   │   ├── preprocess.py                 # 预处理脚本
│   │   ├── preprocessed_data.csv         # 预处理数据（副本）
│   │   └── preprocessed_data_feature_eng.csv # 特征工程数据（副本）
│   │
│   ├── EDA/                      # 探索性数据分析
│   │   ├── eda.py                        # 基础EDA脚本
│   │   ├── eda_map.py                    # 省级交互式地图可视化
│   │   ├── eda_feature_engineering.py    # 特征工程分析
│   │   ├── timeseries_visualization_eda.py # 时间序列可视化EDA
│   │   ├── determine_ranges.py           # 确定指标范围
│   │   ├── Feature_Engineering_EDA/      # 特征工程输出
│   │   ├── Timeseries_EDA/               # 时间序列EDA输出
│   │   └── traditional_eda/              # 传统EDA输出
│   │
│   ├── models/                   # ⭐ 机器学习模型
│   │   │
│   │   ├── pollution_level_classification/  # 污染等级分类模型（推荐）
│   │   │   ├── classification_random_forest.py
│   │   │   ├── classification_xgboost.py
│   │   │   ├── classification_svm.py
│   │   │   ├── classification_logistic_regression.py
│   │   │   ├── classification_catboost.py
│   │   │   ├── classification_lightgbm.py
│   │   │   └── classification_knn.py
│   │   │
│   │   ├── indicator_regression/  # ⭐⭐⭐ 横截面回归预测（论文主要方法）
│   │   │   └── regression_xgboost.py     # XGBoost回归脚本
│   │   │
│   │   ├── timeseries_models_province_revised/ # ⭐ 改进版时间序列（已弃用）
│   │   │   ├── DEPRECATED_ANALYSIS.md    # ⭐⭐⭐ 弃用分析报告 (57KB)
│   │   │   ├── lstm_rf_hybrid_province.py # LSTM+RF混合模型
│   │   │   ├── arima_province_revised.py
│   │   │   ├── sarima_province_revised.py
│   │   │   ├── prophet_province_revised.py
│   │   │   ├── ar_province_revised.py
│   │   │   ├── lstm_province_revised_auto_gpu.py
│   │   │   ├── lstm_province_optimized_auto_gpu.py
│   │   │   └── run_all.py
│   │   │
│   │   ├── archived/              # 归档模型（不再维护）
│   │   │   ├── timeseries_models_station/
│   │   │   └── timeseries_models_station_revised/
│   │   │
│   │   └── backup_prediction_attempts_crosssectional/
│   │       └── ... (实验性脚本)
│   │
│   ├── models_outputs/           # ⭐ 模型输出结果
│   │   ├── indicator_regression/         # ⭐⭐⭐ 回归模型输出
│   │   │   ├── regression_results_summary.csv
│   │   │   ├── regression_report.txt
│   │   │   ├── {indicator}_predictions.png (×7)
│   │   │   ├── {indicator}_feature_importance.png (×7)
│   │   │   └── {indicator}_feature_importance.csv (×7)
│   │   │
│   │   ├── outputs_pollution_level_classification/
│   │   ├── outputs_station_prediction/
│   │   └── outputs_province_prediction/
│   │
│   ├── Outputs/                  # 统一输出目录
│   │   ├── LSTM_RF_Hybrid_Province/      # ⭐ 混合模型输出（失败基线）
│   │   │   ├── hybrid_model_report.txt
│   │   │   └── Province_Plots/
│   │   ├── ARIMA_Province_2023Train_2024Valid_Province/
│   │   ├── SARIMA_Province_2023Train_2024Valid_Province/
│   │   ├── Prophet_Province_2023Train_2024Valid_Province/
│   │   ├── AR_Province_2023Train_2024Valid_Province/
│   │   ├── LSTM_2023Train_2024Valid_Province_GPU/
│   │   ├── LSTM_2023Only_Province_GPU_Optimized/
│   │   ├── TimeSeries_2023Train_2024Valid/
│   │   └── Seasonality_Analysis_SARIMA/
│   │
│   ├── paper_figures/            # ⭐⭐⭐ 论文图表
│   │   ├── data_diagnosis_tri_panel.png       # 数据诊断三联图
│   │   ├── method_comparison_complete.png     # 完整方法对比图
│   │   └── xgboost_regression_comparison.png  # XGBoost性能对比
│   │
│   ├── platform_demo/            # 平台演示（Web应用）
│   │   └── ... (前端/后端代码)
│   │
│   ├── catboost_info/            # CatBoost训练信息
│   │
│   ├── README.md                 # 项目说明文档
│   ├── requirements.txt          # Python依赖列表
│   └── PROJECT_STRUCTURE.md      # 本文档
│
└── 📂 配置文件
    ├── .gitignore                # Git忽略规则
    └── requirements.txt          # Python依赖

```

---

## 🎯 核心目录说明

### 1. `docs/` - 论文和文档 ⭐⭐⭐

**用途**: 存放所有论文相关文档和项目文档

**重要文件**:
- `PAPER_EXPERIMENTS.md` (30KB) - 完整的论文实验报告，包含实验设计、结果模板、论文结构建议
- `PAPER_DELIVERY_SUMMARY.md` - 论文材料交付清单，包含核心发现、论文写作指南
- `FINAL_DELIVERY_COMPLETE.md` - 最终交付完整总结，100%完成标记
- `revise.md` - 修订历史文档

**论文写作时**: 主要参考 `PAPER_EXPERIMENTS.md` 和 `models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md`

### 2. `scripts/` - 分析脚本 ⭐

**用途**: 存放各种临时分析脚本和工具脚本

**主要脚本**:
- `generate_paper_figures.py` - 生成论文图表（数据诊断三联图、方法对比图）
- `seasonality_analysis_sarima.py` - 季节性特征分析
- `run_all.py` - 批量运行时间序列模型
- `run_lstm_rf_hybrid.py` - 运行LSTM+RF混合模型
- `analyze_failure.py` / `deep_analysis.py` / `deep_analysis_v2.py` - 失败分析脚本

**注意**: 这些脚本大多是临时性质，核心模型代码在 `models/` 目录下

### 3. `models/indicator_regression/` - 横截面回归预测 ⭐⭐⭐

**用途**: 论文主要方法，横截面回归预测水质指标

**核心文件**:
- `regression_xgboost.py` (500行) - XGBoost回归模型
  - 26个特征（15水质 + 3工程 + 6时间 + 2空间）
  - 5折分层交叉验证
  - 7个目标指标预测

**性能**:
- 成功指标(4个): R²>0.96 (COD/BOD/Total_N/Total_P)
- 失败指标(3个): R²<0 (DO/Ammonia/pH)

**输出**: `models_outputs/indicator_regression/`

### 4. `models/timeseries_models_province_revised/` - 时间序列模型 ⚠️

**状态**: **已弃用 (DEPRECATED)** ⚠️

**用途**: 作为论文失败基线，展示时间序列方法在不规则采样数据上的失败

**核心文件**:
- `DEPRECATED_ANALYSIS.md` (57KB) - ⭐⭐⭐ 详细的失败分析报告
  - 数据诊断（CV=1.99, ACF≈0）
  - 失败原因理论推导
  - 文献对比
  - 论文写作建议
- `lstm_rf_hybrid_province.py` - LSTM+RF混合模型（已微调，R² ≈ -0.3）

**性能**: 所有时间序列模型R²<0，全部失败

**输出**: `Outputs/LSTM_RF_Hybrid_Province/`

**不推荐用于**: 实际预测或进一步开发

### 5. `models_outputs/indicator_regression/` - 回归模型输出 ⭐⭐⭐

**用途**: XGBoost回归模型的完整实验结果

**文件列表**:
- `regression_results_summary.csv` - 所有指标的交叉验证结果汇总
- `regression_report.txt` - 详细的文字报告
- `{indicator}_predictions.png` (×7) - 预测散点图 + 残差图
- `{indicator}_feature_importance.png` (×7) - 特征重要性可视化
- `{indicator}_feature_importance.csv` (×7) - 特征重要性数据

**论文使用**: 这些文件可直接用于论文Results章节

### 6. `paper_figures/` - 论文图表 ⭐⭐⭐

**用途**: 论文中必须包含的核心图表

**文件列表**:
1. `data_diagnosis_tri_panel.png` - 数据诊断三联图
   - [A] 采样间隔分布 (CV=0.95)
   - [B] ACF图 (ACF≈-0.02)
   - [C] 2023 vs 2024密度对比 (12.4%)

2. `method_comparison_complete.png` - 完整方法对比柱状图
   - 7种方法R²对比（Naive/ARIMA/LSTM/混合/XGBoost回归/分类）
   - 论文核心卖点图

3. `xgboost_regression_comparison.png` - XGBoost性能对比
   - 7个指标的R²和RMSE对比

**论文章节**: Introduction / Methods / Results

---

## 📊 数据流向

```
原始数据 (中国水污染监测数据集.csv)
    ↓
预处理 (preprocess/preprocess.py)
    ↓
预处理数据 (preprocessed_data.csv)
    ↓
特征工程 (EDA/eda_feature_engineering.py)
    ↓
特征工程数据 (preprocessed_data_feature_eng.csv)
    ↓
    ├─→ 时间序列模型 (models/timeseries_models_province_revised/)
    │      → 失败 (R²<0)
    │      → 输出: Outputs/LSTM_RF_Hybrid_Province/
    │      → 论文用途: 失败基线
    │
    ├─→ 横截面回归 (models/indicator_regression/regression_xgboost.py)
    │      → 成功 (4个指标R²>0.96)
    │      → 输出: models_outputs/indicator_regression/
    │      → 论文用途: 主要方法
    │
    └─→ 污染等级分类 (models/pollution_level_classification/)
           → 成功 (准确率99.58%)
           → 输出: models_outputs/outputs_pollution_level_classification/
           → 论文用途: 辅助验证
```

---

## 🎓 论文写作材料索引

### 核心文档（按重要性排序）

| 排名 | 文件 | 位置 | 大小 | 用途 |
|------|------|------|------|------|
| ⭐⭐⭐ | DEPRECATED_ANALYSIS.md | models/timeseries_models_province_revised/ | 57KB | Methods/Results核心材料 |
| ⭐⭐⭐ | PAPER_EXPERIMENTS.md | docs/ | 30KB | 论文整体框架 |
| ⭐⭐ | FINAL_DELIVERY_COMPLETE.md | docs/ | 15KB | 完整总结 + 论点 |
| ⭐⭐ | regression_results_summary.csv | models_outputs/indicator_regression/ | - | Results数据表 |
| ⭐ | PAPER_DELIVERY_SUMMARY.md | docs/ | 15KB | 交付清单 |

### 论文图表（全部必需）

| 图表 | 文件 | 用途 | 状态 |
|------|------|------|------|
| 图1 | paper_figures/data_diagnosis_tri_panel.png | 数据诊断 | ✅ |
| 图2 | paper_figures/method_comparison_complete.png | 方法对比 | ✅ |
| 图3 | models_outputs/indicator_regression/COD_mg_L_predictions.png | 预测结果 | ✅ |
| 图4 | Outputs/LSTM_RF_Hybrid_Province/Province_Plots/*.png | 失败案例 | ✅ |
| 表1 | DEPRECATED_ANALYSIS.md §6.1 | 数据特征对比 | ✅ |
| 表2 | models_outputs/indicator_regression/regression_results_summary.csv | 回归结果 | ✅ |

---

## 🔧 维护说明

### 文件整理规则

1. **临时脚本** → `scripts/`
2. **论文文档** → `docs/`
3. **模型代码** → `models/`
4. **实验输出** → `models_outputs/` 或 `Outputs/`
5. **论文图表** → `paper_figures/`

### 不要在根目录创建

- ❌ 临时Python脚本
- ❌ Markdown文档（除README.md外）
- ❌ 日志文件（nul等）
- ❌ 测试文件

### 保留在根目录

- ✅ README.md
- ✅ requirements.txt
- ✅ PROJECT_STRUCTURE.md
- ✅ 数据集文件（.csv）
- ✅ 配置文件（.gitignore等）

---

## 📝 版本历史

- **v2.5.2** (2025-11-13)
  - 📁 项目结构重组：创建 `docs/` 和 `scripts/` 目录
  - 📊 添加Matplotlib字体优先级规则（黑体优先）
  - 🧹 清理根目录，移除临时文件
  - 📄 创建PROJECT_STRUCTURE.md文档

- **v2.5.1** (2025-11-12)
  - ⭐⭐⭐ 时间序列方法标记为DEPRECATED
  - 📊 完成横截面回归实验
  - 📄 创建完整论文材料文档

- **v2.5** (2025-11-10)
  - ⭐⭐⭐ 新增LSTM+RF混合模型
  - 🔧 Optuna超参数优化

---

**维护者**: Claude Code
**最后更新**: 2025-11-13
**文档版本**: v1.0
