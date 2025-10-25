# 时间序列模型 - 2023训练/2024验证版本

## 📋 概述

这个文件夹包含修改后的时间序列预测模型，专门针对2023/2024年份分割进行了优化。

## ⭐ 主要修改

### 1. 数据分割策略
**原版本**：
- 按时间顺序80/20分割
- 训练集：前80%的数据
- 测试集：后20%的数据

**修改后**：
- 训练集：**仅2023年的数据**
- 测试集：**2024年的数据**
- 目的：解决2024年数据稀疏导致的模型准确率低问题

### 2. 可视化输出
**原版本**：
- 散点图（Actual vs Predicted）
- 适合评估整体预测准确性

**修改后**：
- 折线图（时间序列）
- 蓝色实线：实际值（2024年）
- 红色虚线：预测值
- 更直观展示时间趋势

### 3. 数据筛选
**新增检查**：
- 每个监测站必须同时包含2023年和2024年的数据
- 跳过数据不足的站点

## 📁 文件列表

已修改的模型：
- `arima_station_revised.py` - ARIMA自回归积分滑动平均模型
- 其他模型（通过batch_modify_timeseries_models.py自动生成）

## 🚀 使用方法

### 运行单个模型
```bash
# ARIMA模型
python timeseries_models_station_revised/arima_station_revised.py

# 其他模型同理
python timeseries_models_station_revised/prophet_station_revised.py
```

### 批量生成修改版本
```bash
# 运行批量修改脚本
python batch_modify_timeseries_models.py
```

## 📊 输出结构

每个模型的输出保存在：
```
Outputs/[MODEL_NAME]_2023Train_2024Valid_Station/
├── Station_Plots/              # 各监测站的折线图
│   ├── Province_City_Station1.png
│   ├── Province_City_Station2.png
│   └── ...
└── report.txt                  # 综合分析报告
```

## 📈 预期改进

基于EDA发现：
- ✅ 2023年数据点密集且规则
- ⚠️ 2024年数据点稀疏
- 🎯 使用密集的2023年数据训练，预期模型性能提升

## ⚙️ 技术细节

### 代码修改点
1. **build_timeseries_xxx() 函数**
   ```python
   # 原代码
   split_idx = int(len(data) * 0.8)
   train_data = data[:split_idx]
   test_data = data[split_idx:]

   # 修改后
   data['Year'] = data['Date'].dt.year
   train_data = data[data['Year'] == 2023]
   test_data = data[data['Year'] == 2024]
   test_dates = test_data['Date'].values  # 保存日期
   ```

2. **visualize_station() 函数**
   ```python
   # 原代码（散点图）
   ax.scatter(y_test, y_pred, alpha=0.5, s=30)

   # 修改后（折线图）
   ax.plot(test_dates, y_test, marker='o', label='Actual')
   ax.plot(test_dates, y_pred, marker='s', label='Predicted')
   ```

3. **predict_station() 函数**
   ```python
   # 新增检查
   has_2023 = (station_data['Date'].dt.year == 2023).any()
   has_2024 = (station_data['Date'].dt.year == 2024).any()
   if not (has_2023 and has_2024):
       return None
   ```

## 🔧 故障排查

### 问题1：没有生成任何结果
**可能原因**：监测站缺少2023或2024年的数据
**解决方法**：检查数据集，确认有足够的站点同时包含两年数据

### 问题2：R²仍然很低
**可能原因**：
- 时间序列方法本身不适合不规则采样数据
- 即使分离2023/2024，采样间隔仍然不规则

**建议**：
- 继续使用横截面预测方法（Random Forest、XGBoost）
- 这些方法已证明准确率>99%

## 📝 版本信息

- **创建日期**：2025-10-25
- **作者**：Claude Code
- **版本**：1.0
- **基于**：原timeseries_models_station/目录的模型

## ⚠️ 重要提示

虽然进行了这些修改，但根据项目文档（CLAUDE.md）：
- 时间序列方法仍然不是推荐方法
- 数据采样间隔极不规则是根本问题
- **推荐继续使用横截面分类方法**（XGBoost、Random Forest等）

这些修改版本主要用于：
- 验证2023/2024分割策略的效果
- 对比实验
- 学习和研究目的
