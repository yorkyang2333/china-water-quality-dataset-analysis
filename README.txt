===============================================================================
中国水质污染数据分析项目
===============================================================================

基于中国10个省份180个监测站的真实水质数据（16个指标，15000+条记录），
进行数据分析、机器学习建模和可视化展示。

===============================================================================
项目结构
===============================================================================

china-water-quality-dataset-analysis/
│
├── 【数据文件】
│   ├── 中国水污染监测数据集.csv            # 原始数据
│   ├── preprocessed_data.csv              # 预处理后的数据
│   └── preprocessed_data_feature_eng.csv  # 带工程特征的数据
│
├── preprocess/                            # 数据预处理
│   └── preprocess.py                      # 预处理脚本
│
├── EDA/                                   # 探索性数据分析
│   ├── eda.py                             # 基础EDA
│   ├── eda_map.py                         # 省级交互式地图
│   ├── eda_feature_engineering.py         # 特征工程分析
│   ├── seasonality_analysis_sarima.py     # 季节性分析
│   ├── spatial_distribution_analysis.py   # 空间分布分析
│   ├── temporal_distribution_analysis.py  # 时间分布分析
│   └── bayesian_network_analysis.py       # 贝叶斯网络分析
│
├── models/                                # 机器学习模型
│   ├── pollution_level_classification/    # 污染等级分类（推荐）
│   │   ├── classification_xgboost.py      # XGBoost分类（最佳）
│   │   ├── classification_random_forest.py # 随机森林
│   │   ├── classification_lightgbm.py     # LightGBM
│   │   ├── classification_catboost.py     # CatBoost
│   │   ├── classification_svm.py          # 支持向量机
│   │   ├── classification_knn.py          # K近邻
│   │   └── classification_logistic_regression.py # 逻辑回归
│   │
│   ├── indicator_regression/              # 横截面回归预测（论文主要方法）
│   │   ├── regression_xgboost_clean.py    # XGBoost回归（推荐）
│   │   └── feature_engineering_advanced.py # 高级特征工程
│   │
│   ├── timeseries_models_province_revised/ # 时间序列模型（省份级别，已弃用）
│   │   ├── arima_province_revised.py
│   │   ├── sarima_province_revised.py
│   │   ├── prophet_province_revised.py
│   │   └── lstm_province_revised_auto_gpu.py
│   │
│   └── archived/                          # 归档模型（不再维护）
│
├── SHAP_Analysis_20251119_180422/         # SHAP可解释性分析结果
│
├── docs/                                  # 文档
│   ├── PAPER_MANUSCRIPT_CN.md             # 论文手稿
│   └── SHAP_INTERPRETABILITY_ANALYSIS.md  # SHAP分析总结
│
├── scripts/                               # 分析脚本
│   ├── generate_shap_analysis.py          # 生成SHAP分析
│   └── generate_paper_figures.py          # 生成论文图表
│
├── platform_demo/                         # 平台演示
│   ├── backend/                           # Flask后端
│   └── frontend-vue/                      # Vue 3前端
│
└── Outputs/                               # 模型输出（动态生成）

===============================================================================
快速开始
===============================================================================

1. 环境配置
-----------
pip install -r requirements.txt

主要库：pandas, numpy, scikit-learn, xgboost, catboost, lightgbm, torch,
        shap, matplotlib, seaborn, folium, zhplot, statsmodels, prophet

2. 数据预处理
-------------
cd preprocess
python preprocess.py                       # 生成 preprocessed_data.csv

cd ../EDA
python eda_feature_engineering.py          # 生成 preprocessed_data_feature_eng.csv

3. 探索性数据分析（EDA）
------------------------
cd EDA
python eda.py                              # 基础EDA
python eda_map.py                          # 省级交互式地图
python spatial_distribution_analysis.py    # 空间分布分析
python temporal_distribution_analysis.py   # 时间分布分析
python seasonality_analysis_sarima.py      # 季节性分析
python bayesian_network_analysis.py        # 贝叶斯网络分析

4. 污染等级分类模型（推荐）
---------------------------
cd models/pollution_level_classification
python classification_xgboost.py           # XGBoost分类（最佳，准确率99.58%）
python classification_random_forest.py     # 随机森林（准确率99.55%）
python classification_lightgbm.py          # LightGBM
python classification_catboost.py          # CatBoost

5. 横截面回归预测（论文主要方法）
---------------------------------
cd models/indicator_regression
python regression_xgboost_clean.py         # XGBoost回归（R²=0.97）

6. SHAP可解释性分析
-------------------
cd scripts
python generate_shap_analysis.py           # 生成SHAP分析
python generate_paper_figures.py           # 生成论文图表

7. Web平台演示（可选）
----------------------
cd platform_demo
python start_all.py                        # 一键启动前后端

