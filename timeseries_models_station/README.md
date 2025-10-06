# 时间序列模型 - 按监测站建模

## 📋 概述

本文件夹包含**按监测站独立**的时间序列预测模型。每个监测站独立建模，提供站点级别的精细预测。

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
ar_station_prediction_cpu.py             # CPU标准版
ar_station_prediction_parallel.py        # 多进程并行版
```
- **方法**：AR(p)自回归模型
- **特点**：最简单的时间序列模型
- **建模粒度**：180个监测站 × 16个指标 = 2880个模型

### 🔴 ARIMA模型（AutoRegressive Integrated Moving Average）
```
arima_station_prediction_cpu.py          # CPU标准版
arima_station_prediction_parallel.py     # 多进程并行版（推荐）
```
- **方法**：ARIMA(p,d,q)模型
- **特点**：自动差分、平稳性检验
- **建模粒度**：180个监测站 × 16个指标 = 2880个模型

### 🟢 SARIMA模型（Seasonal ARIMA）
```
sarima_station_prediction_cpu.py         # CPU标准版
sarima_station_prediction_parallel.py    # 多进程并行版（推荐）
```
- **方法**：SARIMA(p,d,q)(P,D,Q,s)季节性模型
- **特点**：考虑季节性因素
- **建模粒度**：180个监测站 × 16个指标 = 2880个模型

### 🟣 LSTM模型（Long Short-Term Memory）
```
lstm_station_prediction_pytorch.py       # PyTorch CPU版（旧版）
lstm_station_prediction_tensorflow.py    # TensorFlow CPU版（旧版）
lstm_station_prediction_gpu.py           # GPU加速版（旧版）
lstm_new_timeseries_analysis_cpu.py      # ⭐ 新版 PyTorch CPU优化版
lstm_new_timeseries_analysis_gpu.py      # ⭐ 新版 PyTorch GPU加速版
```

**新版改进**（推荐使用新版）：
- ✅ 更完善的数据验证
- ✅ 更强的模型架构（128 hidden units, 3层）
- ✅ 学习率调度 + 梯度裁剪
- ✅ 详细的中英文双语报告
- ✅ GPU版本自动适配CUDA/CPU

- **建模粒度**：180个监测站 × 16个指标 = 2880个模型

### 🟡 Prophet模型（Facebook Prophet）
```
prophet_station_prediction_cpu.py        # CPU标准版
prophet_station_prediction_parallel.py   # 多进程并行版
```
- **方法**：Facebook开发的自动时间序列预测
- **特点**：对异常值和缺失值鲁棒
- **建模粒度**：180个监测站 × 16个指标 = 2880个模型

## 🎯 建模方式对比

| 特性 | 按监测站建模（本文件夹） | 按省份建模 |
|------|----------------------|-----------|
| **数据聚合** | 每个监测站独立 | 省内所有监测站聚合 |
| **模型数量** | 180个站 × 16个指标 = 2880个模型 | 10个省份 × 16个指标 = 160个模型 |
| **数据量** | 每个监测站平均86个数据点 | 每个省份平均1552个数据点 |
| **训练时间** | 慢（模型多，数据少） | 快（模型少，数据多） |
| **粒度** | 微观（站级） | 宏观（省级） |
| **用途** | 单站点精细预测 | 省级水质趋势分析 |
| **数据挑战** | 数据点少，更不稳定 | 数据点多，但仍不规则 |

### 为什么监测站建模更困难？
1. **数据量少**：每站平均只有86个点（vs 省份的1552个点）
2. **更不稳定**：单站数据波动大，噪声多
3. **模型多**：2880个模型 vs 160个模型
4. **训练慢**：需要遍历180个监测站

## 📊 输出结果

```
Outputs/
├── AR_Predictions_CPU/                   # AR CPU版输出
├── AR_Predictions_Parallel/              # AR并行版输出
├── ARIMA_Predictions_CPU/                # ARIMA CPU版输出
├── ARIMA_Predictions_Parallel/           # ARIMA并行版输出
├── SARIMA_Predictions_CPU/               # SARIMA CPU版输出
├── SARIMA_Predictions_Parallel/          # SARIMA并行版输出
├── LSTM_Predictions/                     # LSTM旧版输出
├── LSTM_New_CPU/                         # LSTM新版CPU输出
├── LSTM_New_GPU/                         # LSTM新版GPU输出
└── Prophet_Predictions_CPU/              # Prophet输出
```

每个输出目录包含：
- `report.txt` - 文本报告
- `{Province}_{City}_{Station}.png` - 监测站可视化图表
- 180张图表（每个监测站一张）

## 🚀 使用方法

### 1. 基础运行（CPU版本）

```bash
# AR模型
python3 timeseries_models_station/ar_station_prediction_cpu.py

# ARIMA模型
python3 timeseries_models_station/arima_station_prediction_cpu.py

# SARIMA模型
python3 timeseries_models_station/sarima_station_prediction_cpu.py

# LSTM模型（推荐新版）
python3 timeseries_models_station/lstm_new_timeseries_analysis_cpu.py

# Prophet模型
python3 timeseries_models_station/prophet_station_prediction_cpu.py
```

### 2. 加速版本（强烈推荐）

#### 多进程并行版
```bash
# AR并行版
python3 timeseries_models_station/ar_station_prediction_parallel.py

# ARIMA并行版（推荐）
python3 timeseries_models_station/arima_station_prediction_parallel.py

# SARIMA并行版（推荐）
python3 timeseries_models_station/sarima_station_prediction_parallel.py

# Prophet并行版
python3 timeseries_models_station/prophet_station_prediction_parallel.py
```

**说明**：
- 不同监测站并行处理
- 自动检测CPU核心数
- 2-4倍加速

#### GPU加速版（LSTM）
```bash
# LSTM GPU新版（推荐）
python3 timeseries_models_station/lstm_new_timeseries_analysis_gpu.py
```

**要求**：
- NVIDIA GPU + CUDA
- 5-10倍加速

## 📈 性能对比

### 模型速度对比（180个监测站，每站16个指标）

| 模型 | CPU版本 | 加速版本 | 加速倍数 | 预计总时间 |
|------|---------|---------|----------|-----------|
| AR | ~1小时 | ~20分钟 | 3x | 并行版推荐 |
| ARIMA | ~3小时 | ~1小时 | 3x | 并行版推荐 |
| SARIMA | ~6小时 | ~2小时 | 3x | 并行版强烈推荐 |
| LSTM | ~4小时 | ~30-60分钟 | 5-8x | GPU版强烈推荐 |
| Prophet | ~2小时 | ~40分钟 | 3x | 并行版推荐 |

**注**：实际速度取决于硬件配置

### 预期效果（基于实测）

| 模型 | 平均R² | 结论 |
|------|--------|------|
| AR | -0.10 ~ -0.20 | ❌ 效果极差 |
| ARIMA | -0.12 ~ -0.25 | ❌ 效果极差 |
| SARIMA | -0.15 ~ -0.30 | ❌ 效果极差 |
| LSTM | -0.08 ~ -0.22 | ❌ 效果极差 |
| Prophet | -0.10 ~ -0.20 | ❌ 效果极差 |

**结论**：所有监测站级别的时间序列模型效果都很差！

## ✅ 推荐的替代方法

由于时间序列模型效果差，推荐使用：

### 1. 横截面预测（最推荐）
- **位置**：`backup_prediction_attempts_crosssectional/`
- **方法**：Random Forest, XGBoost
- **特征**：经纬度、省份、城市、监测站ID、历史统计

### 2. 省份级建模
- **位置**：`timeseries_models_province/`
- **优势**：数据量更大（1552 vs 86个点）
- **效果**：虽然仍然差，但比监测站级别稍好

### 3. 传统机器学习
- 不依赖时间序列结构
- 利用空间特征和多指标相关性

## 🔍 新版LSTM改进详情

### lstm_new_timeseries_analysis_cpu.py
```python
改进点：
1. 更大的网络（128 units → 64 units 3层）
2. 更完善的数据验证（最小样本数检查）
3. 学习率调度（ReduceLROnPlateau）
4. 更强的正则化（Dropout 0.3）
5. 详细的中英文报告
```

### lstm_new_timeseries_analysis_gpu.py
```python
改进点：
1. GPU加速（自动检测CUDA）
2. 更大的batch size（64）
3. 更多的训练轮数（200 epochs）
4. 自动降级到CPU（无GPU时）
5. 显存优化
```

**但注意**：即使有这些改进，由于时间间隔不规则，效果仍然很差！

## ⚠️ 注意事项

1. **数据要求**
   - 每个监测站最少25-40个数据点
   - 大部分监测站有60-100个数据点
   - 少数监测站数据不足会被跳过

2. **运行时间**
   - 监测站数量多（180个）
   - 强烈推荐使用并行版或GPU版
   - CPU单进程版本训练时间很长

3. **内存需求**
   - CPU版本：2-4GB
   - 并行版本：4-8GB（多进程）
   - GPU版本：2-4GB GPU显存

4. **模型文件大小**
   - 180个监测站的图表：约200-300MB
   - 报告文件：1-2MB

## 📚 相关文档

- **主配置文件**：`../.claude/CLAUDE.md` - 项目整体说明
- **省份模型**：`../timeseries_models_province/README.md` - 按省份建模（数据量更大）
- **横截面预测**：`../backup_prediction_attempts_crosssectional/README.md` - 推荐方法
- **EDA分析**：`../Outputs/EDA/` - 数据特性分析结果

## 🐛 常见问题

### Q1: 为什么训练这么慢？
**A**: 因为有180个监测站，每个站16个指标，共2880个模型。推荐使用并行版或GPU版。

### Q2: 为什么有的监测站被跳过？
**A**: 数据点太少（<25-40个），无法建立有效的时间序列模型。

### Q3: 新版LSTM和旧版有什么区别？
**A**: 新版改进了：
- 网络架构（更深、更宽）
- 数据验证（更严格）
- 学习率调度
- 报告格式（中英文双语）
- GPU自动适配

### Q4: 应该用监测站级还是省份级建模？
**A**:
- ✅ 省份级（数据量大，训练快）
- ⚠️ 监测站级（数据少，训练慢，效果差）
- 但两者效果都不好，推荐用横截面预测

### Q5: 如何只运行部分监测站？
**A**: 修改脚本中的过滤条件：
```python
# 只运行北京的监测站
stations_info = stations_info[stations_info['Province'] == 'Beijing']
```

## 📝 版本历史

### 旧版模型（不推荐）
- `lstm_station_prediction_pytorch.py` - 最早的PyTorch实现
- `lstm_station_prediction_tensorflow.py` - TensorFlow版本
- `lstm_station_prediction_gpu.py` - 早期GPU版本

**问题**：
- 数据验证不够严格
- 报告格式简陋
- 没有学习率调度
- GPU版本不够稳定

### 新版模型（推荐）
- `lstm_new_timeseries_analysis_cpu.py` - 优化的CPU版本
- `lstm_new_timeseries_analysis_gpu.py` - 稳定的GPU版本

**改进**：
- 更完善的数据验证
- 中英文双语报告
- 学习率调度
- GPU自动适配
- 更强的模型架构

## 💡 使用建议

### 如果你一定要运行时间序列模型：

1. **优先级排序**：
   ```
   1. 省份级建模（数据多，训练快）
   2. 监测站级并行版（利用多核CPU）
   3. 监测站级GPU版（如有NVIDIA GPU）
   4. 监测站级CPU单进程版（最慢，不推荐）
   ```

2. **时间规划**：
   ```
   省份级（10省份）：10分钟 ~ 1小时
   监测站级CPU版：1-6小时
   监测站级并行版：20分钟 ~ 2小时
   监测站级GPU版：30-60分钟
   ```

3. **硬件建议**：
   - 最低：4核CPU + 8GB内存
   - 推荐：8核CPU + 16GB内存 或 NVIDIA GPU

### 但再次强调：
⚠️ **时间序列模型效果很差，不推荐用于实际预测！**

建议使用：
- ✅ 横截面预测（Random Forest / XGBoost）
- ✅ 多指标回归
- ✅ 传统机器学习方法

---

**创建日期**: 2025-01-05
**最后更新**: 2025-10-06
**维护者**: 团队项目
**版本**: 2.0（添加新版LSTM说明）
**状态**: ⚠️ 已验证效果差，仅供参考和对比
