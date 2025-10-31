# 🔥🔥🔥 v6.0 快速启动指南

## 一句话总结

**v6.0修复了v5.0的致命错误：把目标指标的历史值放回了输入特征。**

---

## 🚀 立即运行

```bash
cd models/timeseries_models_province_revised
python debug_beijing_v6_fixed_flat.py
```

**运行时间**：约5-10分钟

---

## ✅ 预期结果

### 成功的标志

1. **特征维度显示**：
   ```
   特征维度: 7
   🔥🔥🔥 v6.0关键修复: 目标历史值已纳入特征（含lag1/lag7）
   ```

2. **模型超过baseline**：
   ```
   ✓ 模型超过baseline! (ratio=0.67)
   ```

3. **R² > 0.5**：
   ```
   LSTM模型: R² = 0.62
   结论: LSTM 超过 baseline
   ```

4. **预测图不是平线**：
   - 查看 `Outputs/DEBUG_Beijing_v6_Fixed/prediction_*.png`
   - 橙色预测曲线应该跟随蓝色实际曲线波动

---

## 🔍 核心修复

### v5.0的问题（平线预测）

```python
# ❌ 错误：没有目标的历史值
feature_cols = feature_indicators + ['sin_season', 'cos_season']
```

### v6.0的修复

```python
# ✅ 正确：包含目标 + lag + 其他指标 + 季节
feature_cols = [target_indicator,           # 目标当前值
                f'{target_indicator}_lag1',  # 昨天
                f'{target_indicator}_lag7']  # 7天前
               + feature_indicators          # 其他指标
               + ['sin_season', 'cos_season']
```

---

## 📊 输出文件

| 文件 | 内容 | 重要性 |
|------|------|--------|
| `acf_plots/acf_*.png` | ACF分析图 | 了解时序结构强弱 |
| `prediction_*.png` | 预测对比 | ⭐⭐⭐ 最重要 |
| `training_history_*.png` | 训练曲线 | 检查是否正常收敛 |

---

## 🎯 如何判断成功？

### 场景A：成功（推荐继续）
```
R² > 0.5
预测曲线跟随实际值波动
→ 可以扩展到其他省份和指标
```

### 场景B：中等（需要权衡）
```
0.3 < R² < 0.5
预测有一定波动但不够准
→ 简单的MA7可能更好
```

### 场景C：失败（改用其他方法）
```
R² < 0.3
时序结构太弱
→ 用横截面预测（XGBoost，R² > 99.5%）
```

---

## 🔧 可选调整

### 如果想提升效果

1. **调整滚动窗口**：
   ```python
   # 当前是7日，可以试试3日或14日
   daily_smoothed = daily.rolling(window=3, ...).mean()
   ```

2. **添加更多lag**：
   ```python
   # 添加lag3, lag14等
   daily_smoothed[f'{target_indicator}_lag3'] = ...
   ```

3. **改用StandardScaler**：
   ```python
   # 代替MinMaxScaler
   from sklearn.preprocessing import StandardScaler
   scaler_X = StandardScaler()
   ```

---

## ❓ 常见问题

### Q1: 为什么加目标历史值不算泄漏？

**A**: 窗口是 `[t-seq_len+1, ..., t]` 预测 `t+1`，没有看到未来。

### Q2: 为什么还需要lag特征？

**A**: 显式lag更稳健，LSTM可能学不到所有模式。

### Q3: 如果还是平线怎么办？

**A**: 检查：
1. 特征维度是否 > 5（应该包含目标+lag）
2. ACF最大值是否 < 0.1（说明时序太弱，该换方法）
3. 训练loss是否下降（未下降说明模型有问题）

---

## 📚 相关文档

- **详细说明**: `README_V6_CRITICAL_FIX.md`
- **v5问题分析**: `README_V5_IMPORTANT_FINDINGS.md`
- **项目总览**: `CLAUDE.md`

---

**现在就运行试试吧！** 🚀

```bash
python debug_beijing_v6_fixed_flat.py
```
