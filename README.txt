===============================================================================
中国水质污染数据分析项目 - README
===============================================================================

版本: 2.8 (重构版)
最后更新: 2025-11-22

===============================================================================
项目简介
===============================================================================

这是一个水质污染数据分析与预测项目，涵盖数据预处理、探索性数据分析(EDA)、
机器学习建模和可视化平台演示。项目基于中国10个省份180个监测站的真实水质数据，
包含16个水质指标超过15000条观测记录。

核心研究目标：
- 探索水质污染的时空分布规律
- 构建高精度的污染等级分类模型（准确率 99.5%+）
- 开发水质指标预测模型（横截面回归 R²=0.97）
- 提供可解释性分析（SHAP框架）
- 部署交互式水质监测平台

===============================================================================
核心特点
===============================================================================

✓ 数据量充足
  - 180个监测站，超过15000条观测记录
  - 覆盖10个省份：北京、广东、河南、湖北、江苏、山东、上海、四川、云南、浙江

✓ 指标齐全
  - 16个水质指标：pH、DO、COD、BOD、氨氮、总氮、总磷、重金属等
  - 3个工程特征：重金属总量、COD/BOD比值、氮磷比

✓ 模型丰富
  - 分类模型（17个）：XGBoost、RandomForest、Transformer、终极集成系统等
  - 回归模型：XGBoost回归（7个指标预测）
  - 时间序列模型（已弃用）：LSTM、ARIMA、Prophet等

✓ 可解释性强
  - SHAP分析（4个维度）：全局重要性、特征交互、样本解释、决策路径
  - 贝叶斯网络分析（17条因果关系）
  - 完整的可视化图表（55+ PNG图表）

⚠ 关键问题：时间间隔极不规则
  - 采样间隔变异系数 CV=1.99（远超阈值0.2）
  - 零时间相关性（ACF≈0.03）
  - 2024年数据密度仅为2023年的12.4%
  - 结论：时间序列方法不适合本数据集（R² < 0）

===============================================================================
推荐方法（有效）
===============================================================================

1. 污染等级分类（最推荐）
   - XGBoost分类器：准确率 99.58%
   - RandomForest：准确率 99.55%
   - 终极集成系统：准确率 99.6%+
   - Transformer分类器：准确率 99%+

2. 横截面回归预测（论文主要方法）
   - XGBoost回归（优化版）：R² = 0.97（7个指标平均）
   - 位置：models/indicator_regression/regression_xgboost_clean.py
   - 注意：避免数据泄露版本（improved.py仅适用于缺失值填补）

3. SHAP可解释性分析
   - 分析维度：全局重要性、特征交互、样本解释、决策路径
   - 完成模型：LightGBM分类 + XGBoost回归
   - 输出文件：57个（2个报告 + 55个PNG图表）

🚫 不推荐：时间序列方法
   - LSTM: R² ≈ -0.1（略差于Naive基线）
   - LSTM+RF混合: R² ≈ -0.3
   - 原因：采样极度不规则，零时间相关性
   - 保留目的：作为论文对比基线（失败案例分析）
   - 详细分析：models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md

===============================================================================
项目结构
===============================================================================

china-water-quality-dataset-analysis/
│
├── 数据文件
│   ├── 中国水污染监测数据集.csv          # 原始数据集
│   ├── preprocessed_data.csv            # 预处理后的数据
│   └── preprocessed_data_feature_eng.csv # 带有新工程特征的数据
│
├── preprocess/                          # 数据预处理模块
│   └── preprocess.py                    # 数据预处理脚本
│
├── EDA/                                 # 探索性数据分析
│   ├── eda.py                           # 基础EDA脚本
│   ├── eda_map.py                       # 省级交互式地图可视化
│   ├── eda_feature_engineering.py       # 特征工程分析
│   ├── timeseries_visualization_eda.py  # 时间序列可视化EDA
│   ├── seasonality_analysis_sarima.py   # 季节性特征分析
│   ├── bayesian_network_analysis.py     # 贝叶斯网络分析
│   ├── spatial_distribution_analysis.py # 空间分布分析
│   └── temporal_distribution_analysis.py # 时间分布分析
│
├── models/                              # 机器学习模型
│   │
│   ├── pollution_level_classification/  # 污染等级分类模型（推荐）
│   │   ├── 基础模型（7个）:
│   │   │   ├── classification_random_forest.py      # 随机森林
│   │   │   ├── classification_xgboost.py            # XGBoost（最佳）
│   │   │   ├── classification_svm.py                # 支持向量机
│   │   │   ├── classification_logistic_regression.py # 逻辑回归
│   │   │   ├── classification_catboost.py           # CatBoost
│   │   │   ├── classification_lightgbm.py           # LightGBM
│   │   │   └── classification_knn.py                # K近邻
│   │   │
│   │   └── 高级模型（10个）:
│   │       ├── transformer_classifier.py            # Transformer分类器
│   │       ├── ultimate_ensemble_system.py          # 终极集成系统
│   │       └── ... (其他实验性模型)
│   │
│   ├── indicator_regression/            # 横截面回归预测（论文主要方法）
│   │   ├── regression_xgboost_clean.py          # XGBoost回归（推荐，无泄露）
│   │   ├── regression_xgboost.py                # XGBoost回归（有泄露警告）
│   │   ├── feature_engineering_advanced.py      # 高级特征工程
│   │   └── generate_improvement_comparison.py   # 性能对比生成
│   │
│   ├── timeseries_models_province_revised/ # 改进版时间序列 - Province级别
│   │   ├── arima_province_revised.py
│   │   ├── sarima_province_revised.py
│   │   ├── prophet_province_revised.py
│   │   ├── lstm_province_revised_auto_gpu.py       # 自动GPU加速
│   │   ├── lstm_rf_hybrid_province.py              # LSTM+RF混合模型
│   │   ├── autoformer_province.py                  # Autoformer时序模型
│   │   └── DEPRECATED_ANALYSIS.md                  # ⚠️ 时序方法弃用分析
│   │
│   └── archived/                         # 归档模型（不再维护）
│
├── SHAP_Analysis_20251119_180422/       # SHAP可解释性分析结果
│   ├── Classification_LightGBM/         # 分类模型SHAP分析
│   └── Regression_XGBoost_Regression_Clean/  # 回归模型SHAP分析
│
├── docs/                                # 文档目录
│   ├── PAPER_FRAMEWORK.md              # 论文框架
│   ├── PAPER_MANUSCRIPT_CN.md          # 论文手稿
│   ├── SHAP_INTERPRETABILITY_ANALYSIS.md  # SHAP分析总结
│   ├── analysis_reports/               # 分析报告
│   ├── model_experiments/              # 模型实验记录
│   └── project_management/             # 项目管理文档
│       ├── PROJECT_STRUCTURE.md        # 项目结构说明
│       └── FINAL_DELIVERY_COMPLETE.md  # 论文交付总结
│
├── scripts/                            # 分析脚本
│   ├── generate_paper_figures.py       # 生成论文图表
│   ├── generate_shap_analysis.py       # 生成SHAP分析
│   └── auto_select_best_interpretable_model.py  # 自动选择最佳模型
│
├── platform_demo/                      # 平台演示
│   ├── backend/                        # Flask后端服务
│   │   ├── app.py                      # Flask应用主文件
│   │   ├── models/                     # 模型实现
│   │   └── data/                       # 数据目录
│   └── frontend-vue/                   # Vue 3前端
│
├── Outputs/                            # 模型输出（动态生成）
│
├── .claude/CLAUDE.md                   # 项目配置文档（核心）
├── CHANGELOG.md                        # 版本历史
└── README.txt                          # 本文件

===============================================================================
快速开始
===============================================================================

1. 环境配置
-----------
Python 依赖安装：
  pip install -r requirements.txt

主要库：
  - 数据处理: pandas, numpy
  - 机器学习: scikit-learn, xgboost, catboost, lightgbm
  - 深度学习: torch (PyTorch)
  - 可解释性: shap
  - 可视化: matplotlib, seaborn, folium, zhplot
  - 时间序列: statsmodels, prophet

2. 运行探索性数据分析
---------------------
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

# 空间分布分析
python spatial_distribution_analysis.py

# 时间分布分析
python temporal_distribution_analysis.py

# 贝叶斯网络分析
python bayesian_network_analysis.py

3. 运行分类模型（推荐）
-----------------------
cd models/pollution_level_classification

# 基础模型
python classification_random_forest.py     # 随机森林
python classification_xgboost.py           # XGBoost（最佳）
python classification_catboost.py          # CatBoost
python classification_lightgbm.py          # LightGBM
python classification_svm.py               # SVM
python classification_knn.py               # KNN
python classification_logistic_regression.py # 逻辑回归

# 高级模型
python transformer_classifier.py           # Transformer分类器
python ultimate_ensemble_system.py         # 终极集成系统

4. 运行回归预测（论文方法）
---------------------------
cd models/indicator_regression

python regression_xgboost_clean.py         # XGBoost回归（推荐，无泄露）
python feature_engineering_advanced.py     # 高级特征工程

5. 生成SHAP可解释性分析
-----------------------
cd scripts

python generate_shap_analysis.py           # 生成SHAP分析
python generate_paper_figures.py           # 生成论文图表

6. 运行平台演示（可选）
-----------------------
cd platform_demo

# 一键启动（自动安装依赖 + 启动前后端）
python start_all.py

# 手动启动后端
python run.py

# 手动启动前端（另开终端）
cd frontend-vue
npm install
npm run dev

===============================================================================
模型评估总结
===============================================================================

1. 分类模型（Pollution_Level 预测）
-----------------------------------
模型                     准确率           核心结论
--------------------------------------------------------------------
XGBoost                 99.58%          性能最佳
随机森林                 99.55%          性能与XGBoost非常接近
终极集成系统             99.6%+          超越单一最佳模型
Transformer             99%+            接近XGBoost水平
支持向量机 (SVM)        98.16%          表现良好
逻辑回归                 96.17%          作为线性基准模型

2. 回归模型（指标预测）
-----------------------
方法                           R² (平均)    核心结论
--------------------------------------------------------------------
XGBoost回归（横截面）          0.97         性能优异，适用于本数据集
LSTM时序预测                   -0.1         不适合本数据集（采样不规则）
LSTM+RF混合                    -0.3         不适合本数据集（采样不规则）

总结：对于此数据集，横截面方法（XGBoost、Random Forest）是最佳选择。
时间序列方法因数据不规则而不适用。

===============================================================================
数据泄露问题警告（重要）
===============================================================================

⚠️ 数据泄露问题发现
------------------
在XGBoost回归模型开发过程中，发现了严重的数据泄露（Data Leakage）问题：

问题描述：
- regression_xgboost.py 和 regression_xgboost_improved.py 中的15个高级特征中，
  有11个特征直接使用了预测目标本身或其变换
- 例如：log_Ammonia_N 特征与目标 Ammonia_N_mg_L 的相关系数高达 r=0.996
- 导致模型出现虚假的高性能：R²=0.9993

真实性能：
- 使用无泄露特征的真实性能：R²=-0.037（远低于Naive基线）
- 性能差距：1.036（103倍！）

解决方案：
1. 创建了无泄露版本：regression_xgboost_clean.py
2. 仅使用4个干净特征（不依赖预测目标）
3. 生成完整对比报告：docs/DATA_LEAKAGE_FINAL_REPORT.md（22KB详细分析）

适用场景：
✓ 同时刻缺失值填补：当传感器故障导致某指标缺失，但其他指标正常时，
  可使用 improved 版本（R²=0.9993）
✗ 未来预测：由于未来的目标值未知，无法计算泄露特征，
  必须使用 clean 版本（R²=-0.037）

⚠️ 论文撰写建议
---------------
如果您使用本项目数据进行学术研究，请注意：

1. 必须报告无泄露版本的真实性能（R²=-0.037），而非虚假的0.9993
2. 有泄露版本仅适用于缺失值填补场景，需在论文中明确说明应用场景
3. 时间序列方法失败有系统性证据（ACF=0.03, CV=1.99, 分布漂移），
   可作为失败案例分析
4. 成功的方法：
   - 污染等级分类：准确率 99.5%（XGBoost/RandomForest）
   - 化学污染物回归：部分指标R²可达 0.97（横截面方法）

详细分析请参阅：
- docs/DATA_LEAKAGE_FINAL_REPORT.md - 数据泄露完整报告
- models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md - 时序失败分析
- Outputs/Spatial_Distribution_Analysis/ - 空间分布分析结果
- Outputs/Temporal_Distribution_Analysis/ - 时间分布分析结果

===============================================================================
相关文档
===============================================================================

核心文档：
- .claude/CLAUDE.md - 项目配置文档（最核心）
- CHANGELOG.md - 完整版本历史（v2.0-v2.8）
- docs/PROJECT_STRUCTURE.md - 详细项目结构说明
- docs/PAPER_FRAMEWORK.md - 学术论文框架

分析报告：
- docs/SHAP_INTERPRETABILITY_ANALYSIS.md - SHAP分析总结
- docs/DATA_LEAKAGE_FINAL_REPORT.md - 数据泄露完整报告
- models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md - 时序方法弃用分析
- docs/analysis_reports/ - 各类分析报告

模型文档：
- models/pollution_level_classification/README.md - 分类模型说明
- models/indicator_regression/README.md - 回归模型说明
- models/timeseries_models_province_revised/README.md - 时序模型说明

平台文档：
- platform_demo/README.md - 平台部署说明

===============================================================================
编码规则 & 项目约定
===============================================================================

⚠️ 重要规则（开发者必读）

1. 始终用中文写内容和输出
   - 前端界面、变量名注释、API返回信息全部使用中文
   - 省份显示为中文（北京、广东、河南等），不带"省"字
   - 与用户的所有交互信息必须是中文

2. 不要创建说明文档
   - 不要创建新的 .md 说明文件
   - 所有说明、改动、功能只在 .claude/CLAUDE.md 里记录
   - 代码注释已足够清晰，无需额外文档

3. 模型开发
   - 优化模型放在对应的模型目录中
   - 使用 Optuna 进行超参数优化
   - 新增模型需要在文档中更新

4. Station级别模型归档规则
   - 不再开发和维护 Station级别（监测站级别）的时间序列模型
   - 仅关注 Province级别（省份级别）的时间序列模型
   - 所有Station模型已归档至 models/archived/ 目录

5. 代码中不要使用 Emoji
   - 禁止在 Python 代码的 print 语句中使用 emoji 符号
   - 使用文本标签替代：[OK], [ERROR], [WARN], [INFO], [DONE], [TARGET]
   - 原因：避免 Windows GBK 编码导致的 UnicodeEncodeError

6. Matplotlib 中文字体配置
   - 使用 zhplot 库（推荐）
   - 配置方法：在脚本最开头导入 import zhplot
   - 无需手动配置字体，一行代码解决

7. 优先使用黑体（SimHei）字体
   - 原因：黑体在科学出版物中更常用，符合学术规范
   - zhplot 默认使用黑体，无需额外配置

===============================================================================
技术支持 & 贡献
===============================================================================

问题反馈：
- 请在项目仓库中提交 Issue

贡献指南：
- 欢迎提交 Pull Request
- 请遵循项目编码规则和约定
- 确保代码通过测试

许可证：
- 请根据项目实际情况补充许可证声明（MIT、Apache-2.0 等）

===============================================================================
版本信息
===============================================================================

当前版本: 2.8 (重构版)
创建日期: 2025-01-05
最后更新: 2025-11-22
维护者: Claude Code

完整版本历史请参阅 CHANGELOG.md

===============================================================================
致谢
===============================================================================

感谢所有为本项目做出贡献的开发者和研究者。

祝你使用顺利！

===============================================================================
