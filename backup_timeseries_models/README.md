# 时间序列模型备份文件夹

## ⚠️ 重要说明

**这些时间序列模型不适合当前数据集！**

本文件夹包含所有时间序列预测模型的实现，但由于数据集特性，这些模型的预测效果很差（R² < 0）。

## 📊 数据集问题

### 为什么不适合时间序列预测？

**核心原因：时间间隔极不规则，违反了时间序列模型的基本假设**

1. **时间间隔不规则**（最关键问题）
   - 平均间隔：7-10 天
   - 间隔标准差：17-29 天（变化巨大）
   - 最小间隔：0 天（同日多次采样）
   - 最大间隔：127-222 天（存在大段空白期，超过7个月）
   - **结论**：时间序列模型假设等间隔或近似等间隔采样，本数据严重违反此假设

2. **数据特征不适合时间序列**
   - 更像"不规则抽样"而非"定期时间序列监测"
   - 可能是事件驱动采样（发现污染时才检测）
   - 缺乏可预测的时间依赖结构
   - 大段空白期打断了时间连续性

3. **实测结果证明不可行**
   - 所有时间序列模型 R² < 0（负值）
   - 预测效果比简单平均值还差
   - LSTM、ARIMA、SARIMA、Prophet 全部失效

## 📁 文件列表

### LSTM 模型（深度学习）
- `lstm_station_prediction_pytorch.py` - PyTorch CPU版（旧版）
- `lstm_station_prediction_tensorflow.py` - TensorFlow CPU版（旧版）
- `lstm_station_prediction_gpu.py` - GPU加速版（旧版）
- `lstm_new_timeseries_analysis_cpu.py` - **新版** PyTorch CPU优化版
- `lstm_new_timeseries_analysis_gpu.py` - **新版** PyTorch GPU加速版

### ARIMA 模型（传统统计）
- `arima_station_prediction_cpu.py` - CPU标准版
- `arima_station_prediction_parallel.py` - 多进程并行版

### SARIMA 模型（季节性ARIMA）
- `sarima_station_prediction_cpu.py` - CPU标准版
- `sarima_station_prediction_parallel.py` - 多进程并行版

### Prophet 模型（Facebook开发）
- `prophet_station_prediction_cpu.py` - CPU标准版
- `prophet_station_prediction_parallel.py` - 多进程并行版

### AR 模型（自回归）
- `ar_station_prediction_cpu.py` - CPU标准版
- `ar_station_prediction_parallel.py` - 多进程并行版

## 📈 预期结果

所有这些模型在当前数据集上的表现：
- **R² 平均值**: -0.05 到 -0.15（负值！）
- **结论**: 预测效果比简单均值还差

### 新版LSTM模型改进
新版LSTM脚本（`lstm_new_timeseries_analysis_cpu.py` 和 `lstm_new_timeseries_analysis_gpu.py`）包含以下改进：
- ✅ 更完善的数据验证（最小样本数检查）
- ✅ 更强的模型架构（128 hidden units, 3层LSTM）
- ✅ 学习率调度 + 梯度裁剪
- ✅ 详细的中英文双语报告
- ✅ GPU版本自动适配CUDA/CPU
- ⚠️ **但仍然不适合本数据集！时间间隔不规则是根本问题，无法通过模型优化解决**

## ✅ 推荐方法

本数据集应该使用：
1. **横截面预测** - 利用空间特征（经纬度、省份、城市）
2. **多指标回归** - 用pH、温度、溶解氧等指标预测污染指数
3. **分类模型** - 预测水质污染等级
4. **机器学习** - Random Forest, XGBoost等传统方法

**关键原因**：这些方法不依赖时间序列的连续性和等间隔假设

详见：
- `timeseries_station_prediction.py` - 传统机器学习方法
- `backup_prediction_attempts_crosssectional/` - 横截面预测尝试

## 📚 参考

- 查看 `Outputs/EDA/` 了解数据集的时间序列特性分析
- 查看 `../CLAUDE.md` 了解完整的数据集说明和建议方法

---

**创建日期**: 2025-01-05
**最后更新**: 2025-10-05（修正问题诊断：数据量充足但时间间隔不规则）
**状态**: 已归档（不推荐使用）
**原因**: 时间间隔极不规则（0-222天），违反时间序列模型基本假设
