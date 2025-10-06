# 时间序列模型 - 按省份建模

## 📋 概述

本文件夹包含**按省份聚合**的时间序列预测模型。与按监测站建模不同，这里将每个省份内所有监测站的数据聚合后建模，提供更宏观的省级预测。

## ⚠️ 重要警告

**这些时间序列模型不适合当前数据集！**

### 核心问题：时间间隔极不规则

| 问题 | 数据表现 | 影响 |
|------|---------|------|
| **间隔不规则** | 0-222天的巨大差异 | 模型无法学习稳定的时间模式 |
| **大段空白** | 最长空白期超过7个月 | 打断了时间依赖关系 |
| **实测R² < 0** | 所有模型负R² | 预测效果比简单平均值还差 |

**结论**：数据更像"不规则抽样"而非"定期时间序列监测"，违反了时间序列模型的基本假设。

### 为什么保留这些模型？
- 📚 **教育目的**：展示模型实现和失败原因
- 🔬 **对比实验**：与横截面预测对比，证明数据特性的重要性
- 📊 **完整记录**：记录所有尝试过的方法

## 📁 文件列表

### 🔵 AR模型（AutoRegressive）
```
ar_province_prediction.py
```
- **方法**：AR(p)自回归模型
- **特点**：最简单的时间序列模型，只考虑历史值的线性组合
- **参数**：自动选择滞后阶数p（基于AIC准则）
- **速度**：快
- **建模粒度**：10个省份（聚合省内所有监测站）

### 🔴 ARIMA模型（AutoRegressive Integrated Moving Average）
```
arima_province_prediction.py             # CPU版本（单进程）
arima_province_prediction_parallel.py    # 多进程并行加速版（推荐）
```
- **方法**：ARIMA(p,d,q)自回归积分滑动平均模型
- **特点**：自动差分、平稳性检验（ADF test）
- **参数**：自动选择(p,d,q)参数（基于AIC准则）
- **速度**：
  - CPU版：中等
  - 并行版：**2-4倍加速**（利用多核CPU）
- **建模粒度**：10个省份

**选择建议**：
- ✅ 多核CPU（4核+）→ 使用 `arima_province_prediction_parallel.py`
- 单核/双核CPU → 使用 `arima_province_prediction.py`

### 🟢 SARIMA模型（Seasonal ARIMA）
```
sarima_province_prediction.py            # CPU版本（单进程）
sarima_province_prediction_parallel.py   # 多进程并行加速版（推荐）
```
- **方法**：SARIMA(p,d,q)(P,D,Q,s)季节性ARIMA模型
- **特点**：考虑季节性因素（月度周期12或季度周期4）
- **参数**：自动选择参数（基于AIC准则）
- **速度**：
  - CPU版：慢（参数搜索空间大）
  - 并行版：**2-4倍加速**
- **建模粒度**：10个省份

**选择建议**：
- ✅ 强烈推荐使用并行版本（SARIMA参数多，训练慢）

### 🟣 LSTM模型（Long Short-Term Memory）
```
lstm_province_prediction.py              # PyTorch CPU版本
lstm_province_prediction_gpu.py          # PyTorch GPU加速版（推荐）
```
- **方法**：3层LSTM神经网络（64 hidden units）
- **框架**：PyTorch
- **特点**：
  - 序列长度：30个时间步
  - 数据归一化（MinMaxScaler）
  - 早停策略（patience=15）
  - 梯度裁剪（max_norm=1.0）
- **速度**：
  - CPU版：慢（深度学习模型）
  - GPU版：**5-10倍加速**（需NVIDIA GPU + CUDA）
- **建模粒度**：10个省份

**选择建议**：
- ✅ 有NVIDIA GPU → 使用 `lstm_province_prediction_gpu.py`
- 无GPU → 使用 `lstm_province_prediction.py`（但训练很慢）

**GPU版本特性**：
- 自动检测GPU可用性
- 无GPU时自动降级到CPU
- 更大的batch size（64 vs 32）
- 更高的学习率（0.001 vs 0.0005）
- 更多的训练轮数（200 vs 150）

### 🟡 Prophet模型（Facebook Prophet）
```
prophet_province_prediction.py
```
- **方法**：Facebook开发的自动时间序列预测
- **特点**：
  - 自动检测趋势和季节性
  - 对异常值和缺失值鲁棒
  - 乘法季节性模型
- **速度**：快（优化的Stan后端）
- **建模粒度**：10个省份

## 🎯 建模方式对比

| 特性 | 按省份建模（本文件夹） | 按监测站建模 |
|------|---------------------|-------------|
| **数据聚合** | 省内所有监测站聚合 | 每个监测站独立 |
| **模型数量** | 10个省份 × 16个指标 = 160个模型 | 180个站 × 16个指标 = 2880个模型 |
| **数据量** | 每个省份平均1552个数据点 | 每个监测站平均86个数据点 |
| **训练时间** | 快（模型少） | 慢（模型多） |
| **粒度** | 宏观（省级） | 微观（站级） |
| **用途** | 省级水质趋势分析 | 单站点精细预测 |

## 📊 输出结果

每个模型的输出保存在 `Outputs/` 目录下：

```
Outputs/
├── AR_Province_Predictions/              # AR省份模型输出
│   ├── report.txt                        # 文本报告
│   ├── Beijing.png                       # 北京省份可视化
│   ├── Shanghai.png                      # 上海省份可视化
│   └── ...                               # 其他省份
│
├── ARIMA_Province_Predictions/           # ARIMA CPU版输出
├── ARIMA_Province_Predictions_Parallel/  # ARIMA并行版输出
├── SARIMA_Province_Predictions/          # SARIMA CPU版输出
├── SARIMA_Province_Predictions_Parallel/ # SARIMA并行版输出
├── LSTM_Province_Predictions/            # LSTM CPU版输出
├── LSTM_Province_Predictions_GPU/        # LSTM GPU版输出
└── Prophet_Province_Predictions/         # Prophet输出
```

### 报告内容
每个 `report.txt` 包含：
1. **总体性能统计**（R²、RMSE、MAE）
2. **按省份统计**（各省份的平均R²）
3. **详细结果**（每个省份×指标的性能，按R²排序）
4. **模型参数分布**（ARIMA/SARIMA参数统计）

### 可视化图表
每个省份生成一张图表，包含16个子图（每个指标一个），展示：
- 实际值 vs 预测值的散点图
- 完美预测线（红色虚线）
- 性能指标（R²、RMSE、样本数）

## 🚀 使用方法

### 1. 基础运行（CPU版本）
```bash
# AR模型
python3 timeseries_models_province/ar_province_prediction.py

# ARIMA模型
python3 timeseries_models_province/arima_province_prediction.py

# SARIMA模型
python3 timeseries_models_province/sarima_province_prediction.py

# LSTM模型
python3 timeseries_models_province/lstm_province_prediction.py

# Prophet模型
python3 timeseries_models_province/prophet_province_prediction.py
```

### 2. 加速版本（推荐）

#### 多进程并行版（ARIMA/SARIMA）
```bash
# ARIMA并行版（2-4倍加速）
python3 timeseries_models_province/arima_province_prediction_parallel.py

# SARIMA并行版（2-4倍加速）
python3 timeseries_models_province/sarima_province_prediction_parallel.py
```

**说明**：
- 自动检测CPU核心数
- 不同指标并行训练
- 适合多核CPU（4核及以上）

#### GPU加速版（LSTM）
```bash
# LSTM GPU版（5-10倍加速）
python3 timeseries_models_province/lstm_province_prediction_gpu.py
```

**要求**：
- NVIDIA GPU + CUDA
- PyTorch with CUDA support
- 无GPU时自动降级到CPU

### 3. 环境配置
```bash
# 安装基础依赖
pip install pandas numpy scikit-learn matplotlib seaborn
pip install statsmodels prophet

# ARIMA/SARIMA需要
pip install statsmodels

# Prophet需要
pip install prophet

# LSTM CPU版需要
pip install torch

# LSTM GPU版需要（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 性能对比

### 模型速度对比（单个省份，16个指标）

| 模型 | CPU版本 | 加速版本 | 加速倍数 | 硬件要求 |
|------|---------|---------|----------|---------|
| AR | ~30秒 | - | - | CPU |
| ARIMA | ~2分钟 | ~30秒 | 4x | 多核CPU |
| SARIMA | ~5分钟 | ~1.5分钟 | 3-4x | 多核CPU |
| LSTM | ~10分钟 | ~1-2分钟 | 5-10x | NVIDIA GPU |
| Prophet | ~1分钟 | - | - | CPU |

**注**：实际速度取决于硬件配置和数据量

### 预期效果（基于实测）

| 模型 | 平均R² | 结论 |
|------|--------|------|
| AR | -0.05 ~ -0.15 | ❌ 效果极差 |
| ARIMA | -0.08 ~ -0.20 | ❌ 效果极差 |
| SARIMA | -0.10 ~ -0.25 | ❌ 效果极差 |
| LSTM | -0.05 ~ -0.18 | ❌ 效果极差 |
| Prophet | -0.06 ~ -0.16 | ❌ 效果极差 |

**结论**：所有时间序列模型R² < 0（负值），预测效果比简单均值还差！

## ✅ 推荐的替代方法

由于时间序列模型效果差，推荐使用：

### 1. 横截面预测（最推荐）
- **位置**：`backup_prediction_attempts_crosssectional/`
- **方法**：Random Forest, XGBoost, Gradient Boosting
- **特征**：经纬度、省份、城市、历史统计值
- **优势**：不依赖时间序列的连续性

### 2. 多指标回归
- **方法**：用其他水质指标预测目标指标
- **特征**：pH、温度、溶解氧、浊度、重金属等
- **优势**：利用指标间的物理化学相关性

### 3. 传统机器学习
- **方法**：SVM、随机森林、梯度提升树
- **优势**：适合不规则采样数据

## 📚 相关文档

- **主配置文件**：`../.claude/CLAUDE.md` - 项目整体说明
- **监测站模型**：`../timeseries_models_station/README.md` - 按监测站建模
- **横截面预测**：`../backup_prediction_attempts_crosssectional/README.md` - 推荐方法
- **EDA分析**：`../Outputs/EDA/` - 数据特性分析结果

## 🔍 技术细节

### 数据预处理
所有模型使用相同的预处理流程：
1. 按省份聚合数据
2. 按时间排序
3. 数据清洗（去除缺失值）
4. 时间顺序分割（80% 训练，20% 测试）

### 模型参数

#### AR模型
- 滞后阶数：1-10（自动选择）
- 选择准则：AIC（Akaike Information Criterion）

#### ARIMA模型
- p（AR阶数）：0-3
- d（差分阶数）：0-2
- q（MA阶数）：0-3
- 选择准则：AIC

#### SARIMA模型
- 基础参数：(p,d,q) ∈ [0,3] × [0,2] × [0,3]
- 季节参数：(P,D,Q,s) ∈ [0,2] × [0,2] × [0,2] × {4,12}
- 季节周期：12（月度）或4（季度）
- 选择准则：AIC

#### LSTM模型
- 网络结构：3层LSTM，64 hidden units
- 序列长度：30个时间步
- Dropout：0.3
- 学习率：0.0005（CPU）/ 0.001（GPU）
- Batch size：32（CPU）/ 64（GPU）
- Optimizer：Adam with weight decay (1e-5)

#### Prophet模型
- 季节性：年度、月度/季度
- 季节性模式：乘法模型
- 变化点先验：0.05（降低过拟合）
- 置信区间：95%

## ⚠️ 注意事项

1. **数据要求**
   - AR：最少25个数据点
   - ARIMA：最少30个数据点
   - SARIMA：最少40个数据点
   - LSTM：最少60个数据点（seq_length=30）
   - Prophet：最少30个数据点

2. **内存需求**
   - CPU版本：< 2GB
   - 并行版本：2-4GB（多进程）
   - GPU版本：2-4GB GPU显存

3. **运行时间**
   - 省份模型：10个省份 × 16个指标 = 160个模型
   - 预计总时间：10分钟（AR）~ 1小时（SARIMA CPU版）
   - 使用加速版本可显著减少时间

4. **GPU版本注意**
   - 需要NVIDIA GPU
   - 需要安装CUDA和cuDNN
   - 确保PyTorch支持CUDA
   - 检查：`python -c "import torch; print(torch.cuda.is_available())"`

## 🐛 常见问题

### Q1: 为什么R²是负值？
**A**: 负R²说明模型预测效果比简单均值还差，证明时间序列方法不适合本数据集。

### Q2: GPU版本报错找不到CUDA？
**A**: 检查：
1. 是否有NVIDIA GPU
2. CUDA是否正确安装
3. PyTorch是否支持CUDA版本

### Q3: 并行版本为什么内存占用高？
**A**: 多进程会复制数据到每个进程，可以减少进程数：
```python
# 在脚本中修改
n_processes = 2  # 减少进程数
```

### Q4: 如何只运行部分省份？
**A**: 修改脚本中的省份列表：
```python
provinces = ['Beijing', 'Shanghai']  # 只运行这两个省份
```

## 📝 贡献指南

如果要添加新的时间序列模型：

1. **文件命名**：`{model}_province_prediction_{version}.py`
2. **输出目录**：`Outputs/{ModelName}_Province_Predictions_{Version}/`
3. **包含内容**：
   - 模型实现
   - 性能评估（R²、RMSE、MAE）
   - 可视化图表
   - 文本报告
4. **文档更新**：更新本README.md

---

**创建日期**: 2025-10-06
**最后更新**: 2025-10-06
**维护者**: 团队项目
**版本**: 1.0
**状态**: ⚠️ 已验证效果差，仅供参考和对比
