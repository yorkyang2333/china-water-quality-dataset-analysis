# 项目文档目录 (docs/)

本目录包含水质污染数据分析项目的所有文档，按主题分类组织。

---

## 📁 目录结构

```
docs/
├── README.md                          # 本文件
├── SHAP_INTERPRETABILITY_ANALYSIS.md  # ⭐⭐⭐ SHAP可解释性完整分析报告（最新）
│
├── analysis_reports/                  # 分析报告
│   ├── DATA_LEAKAGE_FINAL_REPORT.md  # 数据泄露问题最终报告
│   └── IMPROVEMENT_SUMMARY.md        # 模型改进总结
│
├── model_experiments/                 # 模型实验文档
│   ├── PAPER_EXPERIMENTS.md          # 论文实验记录
│   └── revise.md                     # 模型修订建议
│
├── project_management/                # 项目管理文档
│   ├── FINAL_DELIVERY_COMPLETE.md    # 最终交付完成报告
│   ├── PAPER_DELIVERY_SUMMARY.md     # 论文材料交付总结
│   ├── PROGRESS_REPORT_20251114.md   # 进度报告（2025-11-14）
│   └── PROJECT_STRUCTURE.md          # 项目结构说明
│
└── data_documentation/                # 数据文档
    └── 中国水污染监测数据集说明.docx  # 原始数据集说明文档
```

---

## 📖 核心文档

### 🌟 最新报告

#### [SHAP_INTERPRETABILITY_ANALYSIS.md](./SHAP_INTERPRETABILITY_ANALYSIS.md)
**生成日期**: 2025-11-19
**类型**: 可解释性分析完整报告
**内容概要**:
- 自动模型选择（LightGBM分类 + XGBoost回归）
- 全局特征重要性分析（Top 5关键因素）
- 特征交互分析（COD×浊度、总氮×NP比等）
- 样本级别解释（36+3个典型案例）
- 决策路径可视化
- 业务应用建议
- 完整技术附录

**适用场景**:
- ✅ 论文写作（可解释性机器学习章节）
- ✅ 项目汇报（向管理层展示）
- ✅ 业务应用（监测站点优化、预警系统设计）
- ✅ 公众沟通（科普教育材料）

**亮点**:
- 📊 58个可视化图表
- 📈 多层次分析（全局→交互→局部→路径）
- 🎯 可操作的业务建议
- 📚 完整的方法论说明

---

## 📂 分类浏览

### 1️⃣ 分析报告 (analysis_reports/)

#### DATA_LEAKAGE_FINAL_REPORT.md
**日期**: 2025-11-14
**主题**: 数据泄露问题
**关键内容**:
- 发现11个泄露特征（与预测目标相关性>0.7）
- 有泄露版R²=0.9993（虚假高性能）
- 无泄露版R²=-0.037（真实性能）
- 泄露检测checklist

#### IMPROVEMENT_SUMMARY.md
**日期**: 2025-11-14
**主题**: 模型改进总结
**关键内容**:
- XGBoost回归实验（7个目标指标）
- 横截面vs时间序列方法对比
- 改进建议

---

### 2️⃣ 模型实验 (model_experiments/)

#### PAPER_EXPERIMENTS.md
**日期**: 2025-11-13
**主题**: 论文实验设计
**关键内容**:
- 分类模型（XGBoost/RandomForest, 准确率99.5%）
- 回归模型（化学污染物R²=0.97，生物指标R²<0）
- 时间序列模型（LSTM R²=-0.1，已弃用）
- 完整的实验参数和结果

#### revise.md
**日期**: 2025-11-12
**主题**: 混合模型修订建议
**关键内容**:
- LSTM+RF混合模型的问题诊断
- 三大关键补丁（统一评估切片、重训最终模型、带偏置岭回归）
- 数据分布漂移对策

---

### 3️⃣ 项目管理 (project_management/)

#### FINAL_DELIVERY_COMPLETE.md
**日期**: 2025-11-13
**主题**: 论文材料100%完成
**关键内容**:
- XGBoost回归实验完成（7个指标）
- 论文图表生成（3个核心图）
- 完整交付清单

#### PAPER_DELIVERY_SUMMARY.md
**日期**: 2025-11-13
**主题**: 论文材料交付总结
**关键内容**:
- 成功方法：分类99.5%、化学污染物回归R²=0.97
- 失败方法：时间序列R²<0（有系统性证据）
- 论文建议结构

#### PROJECT_STRUCTURE.md
**日期**: 2025-11-13
**主题**: 项目结构说明
**关键内容**:
- 完整的目录树
- 各模块功能说明
- 文件命名规范

#### PROGRESS_REPORT_20251114.md
**日期**: 2025-11-14
**主题**: 项目进度报告
**关键内容**:
- EDA功能新增（时空分布、贝叶斯网络）
- 数据泄露问题修复
- 编码规则更新

---

### 4️⃣ 数据文档 (data_documentation/)

#### 中国水污染监测数据集说明.docx
**日期**: 2025-10-27
**主题**: 原始数据集说明
**关键内容**:
- 数据来源和采集方法
- 指标定义和单位
- 数据质量说明

---

## 🔍 按需查找

### 按任务查找

| 任务 | 推荐文档 |
|:-----|:---------|
| **写论文** | SHAP_INTERPRETABILITY_ANALYSIS.md, PAPER_EXPERIMENTS.md |
| **做汇报** | SHAP_INTERPRETABILITY_ANALYSIS.md, FINAL_DELIVERY_COMPLETE.md |
| **优化模型** | analysis_reports/IMPROVEMENT_SUMMARY.md, model_experiments/revise.md |
| **理解数据** | data_documentation/中国水污染监测数据集说明.docx |
| **了解进度** | project_management/PROGRESS_REPORT_20251114.md |
| **解决问题** | analysis_reports/DATA_LEAKAGE_FINAL_REPORT.md |

### 按时间查找

| 日期 | 文档 | 主题 |
|:-----|:-----|:-----|
| **2025-11-19** | **SHAP_INTERPRETABILITY_ANALYSIS.md** | **可解释性完整分析** ⭐ |
| 2025-11-14 | DATA_LEAKAGE_FINAL_REPORT.md | 数据泄露修复 |
| 2025-11-14 | PROGRESS_REPORT_20251114.md | 进度报告 |
| 2025-11-14 | IMPROVEMENT_SUMMARY.md | 模型改进 |
| 2025-11-13 | FINAL_DELIVERY_COMPLETE.md | 论文完成 |
| 2025-11-13 | PAPER_DELIVERY_SUMMARY.md | 交付总结 |
| 2025-11-13 | PROJECT_STRUCTURE.md | 结构说明 |
| 2025-11-13 | PAPER_EXPERIMENTS.md | 实验记录 |
| 2025-11-12 | revise.md | 修订建议 |
| 2025-10-27 | 中国水污染监测数据集说明.docx | 数据说明 |

---

## 📝 文档规范

### 命名规范

- **全大写下划线**: 重要报告（如`SHAP_INTERPRETABILITY_ANALYSIS.md`）
- **日期后缀**: 定期报告（如`PROGRESS_REPORT_20251114.md`）
- **主题前缀**: 专题文档（如`DATA_LEAKAGE_FINAL_REPORT.md`）

### 格式规范

- **标题层级**: # 一级（文档标题）→ ## 二级（章节）→ ### 三级（小节）
- **代码块**: 使用```语言标识符```
- **表格**: 使用Markdown表格格式
- **强调**: **重要内容**加粗，*次要内容*斜体
- **列表**: 有序用数字，无序用-或*

### 内容规范

- **开头**: 包含生成日期、作者、版本号
- **目录**: 超过3页的文档需包含目录
- **摘要**: 长文档需包含执行摘要
- **引用**: 引用其他文档时使用相对路径
- **结尾**: 包含附录、术语表、相关资源

---

## 🔗 相关资源

### 项目根目录

- **数据文件**: `../preprocessed_data_feature_eng.csv`
- **脚本目录**: `../scripts/`
- **模型目录**: `../models/`
- **输出目录**: `../Outputs/`

### 在线资源

- **SHAP官方文档**: https://shap.readthedocs.io/
- **LightGBM文档**: https://lightgbm.readthedocs.io/
- **XGBoost文档**: https://xgboost.readthedocs.io/

---

## 📧 文档维护

### 更新记录

| 日期 | 更新内容 | 更新人 |
|:-----|:---------|:-------|
| 2025-11-19 | 创建docs/README.md，整理文档结构 | Claude Code |
| 2025-11-19 | 新增SHAP_INTERPRETABILITY_ANALYSIS.md | Claude Code |
| 2025-11-14 | 新增DATA_LEAKAGE_FINAL_REPORT.md | Claude Code |

### 维护规则

1. **新文档**: 放入对应子目录，更新本README
2. **旧文档**: 超过6个月的文档移至`archive/`子目录
3. **版本控制**: 重要文档修改时保留旧版本（添加v1/v2后缀）

---

**最后更新**: 2025-11-19
**维护者**: Claude Code
**文档版本**: v1.0
