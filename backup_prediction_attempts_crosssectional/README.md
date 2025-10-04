# 预测尝试备份

此文件夹包含之前的多次预测尝试，这些方法由于数据特性问题效果不理想。

## 文件列表

1. **gtwr_with_accuracy_evaluation.py** (已废弃)
   - 使用 GTWR (Geographically and Temporally Weighted Regression) 模型
   - 问题：经纬度数据有误，时空加权不准确
   - 结果：R² ≈ 0.15-0.25

2. **gtwr_prediction_by_province.py** (已废弃)
   - 改进版GTWR，使用城市标准坐标
   - 问题：仍然受限于数据质量，预测效果差
   - 结果：R² ≈ 0.15-0.25

3. **improved_province_prediction.py** (已废弃)
   - 使用 RandomForest, GradientBoosting, Ridge 等多种模型
   - 问题：数据混合了不同站点，时间序列不连续
   - 结果：R² ≈ 0.15

4. **station_based_prediction.py** (已废弃)
   - 按监测站分组建模，包含滞后特征
   - 问题：数据采样稀疏，自相关性极低
   - 结果：R² ≈ 0.25

5. **crosssectional_prediction.py** (已废弃)
   - 横截面预测，使用其他指标互相预测
   - 问题：指标间相关性极低（< 0.05）
   - 结果：R² < 0

6. **final_cleaned_prediction.py** (已废弃)
   - 增加严格数据清洗，过滤异常值
   - 问题：清洗后指标仍然独立
   - 结果：R² < 0

7. **baseline_prediction.py** (参考)
   - 基线模型（历史均值法）
   - 说明：当数据无规律时的标准方法
   - 结果：R² ≈ 0

## 问题总结

所有这些尝试失败的根本原因：

1. **数据质量问题**
   - 大量异常值（水温-61°C, pH 199等）
   - 指标间相关性极低（< 0.05）
   - 时间序列无自相关

2. **数据特性**
   - 各指标似乎是独立随机生成的
   - 缺乏物理/化学上应有的关系
   - 不满足机器学习建模的前提

3. **采样问题**
   - 时间间隔不规律
   - 不同站点数据混合
   - 缺乏真正的时间连续性

## 最终方案

使用 `timeseries_station_prediction.py`（主目录）：
- 纯时间序列分析
- 每个监测站独立建模
- 只使用历史值和时间特征
- 预期 R² 0.3-0.7

---

**保留这些文件的目的**：记录探索过程，展示不同方法的尝试。
