# 🚀 v7.0 快速启动指南

## 一句话总结

**v7.0 在 v6.0 的基础上，增加了外生因子lag、残差连接、更强baseline、AMP加速和RobustScaler。**

---

## 🔥 立即运行

```bash
cd models/timeseries_models_province_revised
python debug_beijing_v7_enhanced.py
```

**运行时间**：
- 无GPU：~20-30分钟
- 有GPU（CUDA）：~10-15分钟（AMP加速）

---

## ✅ 预期结果

### 成功的标志

1. **特征维度大幅增加**：
   ```
   🔥 v7.0 特征增强: 目标3列 + 外生12列 + 季节2列 = 17列
   ```
   （比v6.0的7列增加了10列）

2. **3个baseline都显示**：
   ```
   Naive baseline MSE: 0.012
   MA7 baseline MSE: 0.015
   🔥 v7.0 Seasonal Naive MSE: 0.010
   ```

3. **残差连接生效**：
   - 模型输出 = last_target + delta
   - 预测增量而非绝对值

4. **R² 进一步提升**：
   ```
   v6.0: R² = 0.62
   v7.0: R² = 0.67  （提升 0.05）
   ```

5. **AMP加速（GPU）**：
   ```
   ✓ 使用CUDA GPU: NVIDIA RTX 3090
   🔥 v7.0: cuDNN benchmark enabled
   ```

---

## 🔍 核心改进（vs v6.0）

### 改进1：外生因子也添加 lag/ma7

```python
# v6.0 ❌：只给目标加 lag
feature_cols = [target, f'{target}_lag1', f'{target}_lag7'] + feature_indicators

# v7.0 ✅：所有外生因子都加 lag1/lag7/ma7
for col in feature_indicators:
    daily[f'{col}_lag1'] = daily[col].shift(1)
    daily[f'{col}_lag7'] = daily[col].shift(7)
    daily[f'{col}_ma7']  = daily[col].rolling(7, min_periods=1).mean()
```

**为什么有效**：
- 水温昨天的值影响今天的DO（相位差）
- pH的7日均值比瞬时值更稳定

---

### 改进2：残差连接

```python
# v6.0 ❌：直接预测绝对值
def forward(self, x):
    out = self.lstm(x)
    return self.head(out)

# v7.0 ✅：预测增量 Δ
def forward(self, x):
    last_target = x[:, -1, self.target_col_idx].unsqueeze(-1)  # y[t]
    delta = self.head(self.lstm(x))                            # Δ
    return last_target + delta                                  # y[t] + Δ
```

**为什么有效**：
- 降低预测难度（预测相对变化 vs 绝对值）
- 强持久性时，模型只需预测小的 Δ

---

### 改进3：Seasonal Naive baseline

```python
# v6.0 ❌：只有2个baseline
Naive: y[t] = y[t-1]
MA7:   y[t] = mean(y[-7:])

# v7.0 ✅：新增第3个baseline
Seasonal: y[t] = y[t-7]
```

**为什么重要**：
- 对周模式数据（如水温），Seasonal常最强
- LSTM至少要超过最佳baseline

---

### 改进4：ReduceLROnPlateau + AMP

```python
# v6.0 ❌：固定周期降低学习率
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# v7.0 ✅：基于验证损失自适应调整
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.5, patience=4)

# 🔥 AMP 混合精度（GPU加速30-50%）
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
    loss = criterion(model(X), y)
scaler.scale(loss).backward()
```

---

### 改进5：RobustScaler

```python
# v6.0 ❌：所有特征都用 MinMaxScaler
scaler_X = MinMaxScaler()

# v7.0 ✅：分别缩放
scaler_X = RobustScaler()    # Features（应对离群值）
scaler_y = MinMaxScaler()    # Target（保持[0,1]）
```

---

## 📊 输出文件

| 文件 | 内容 | v7.0新增 |
|------|------|----------|
| `acf_plots/acf_*.png` | ACF分析图 | - |
| `prediction_*.png` | 预测对比 | ✅ 标注v7.0增强 |
| 控制台输出 | 3个baseline对比 | ✅ Seasonal Naive |

---

## 🎯 如何判断v7.0比v6.0更好？

### 场景A：明显提升（推荐继续用v7.0）
```
v6.0: R² = 0.62
v7.0: R² = 0.68
提升：0.06 → ✅ 外生因子lag生效
```

### 场景B：小幅提升（v7.0更稳）
```
v6.0: R² = 0.65
v7.0: R² = 0.67
提升：0.02 → ✅ 残差连接+RobustScaler提升鲁棒性
```

### 场景C：提升不明显（时序结构太弱）
```
v6.0: R² = 0.35
v7.0: R² = 0.37
提升：0.02 → ⚠️ 考虑其他方法（LightGBM/横截面）
```

---

## 🔧 可选调整

### 如果想进一步提升

1. **减少外生因子**（如果>5个）：
   ```python
   # 只保留最相关的2-3个
   feature_indicators = ['Water_Temperature_C', 'pH']
   ```

2. **调整滚动窗口**：
   ```python
   # 当前7日，可试3日或14日
   daily_smoothed = daily.rolling(window=3, ...).mean()
   ```

3. **试用LightGBM对比**：
   ```python
   # 用相同的lag/ma7特征训练树模型
   # 如果显著更好 → Stacking
   ```

---

## ❓ 常见问题

### Q1: v7.0 会更慢吗？

**A**: 取决于硬件
- 无GPU：稍慢（~10%），因为特征多了
- 有GPU：更快（AMP加速超过特征增加的开销）

### Q2: 为什么特征从7列变成17列？

**A**: 外生因子也加了lag/ma7
```
v6.0: 目标3列 + 外生2列 + 季节2列 = 7列
v7.0: 目标3列 + 外生(2×4=8)列 + 季节2列 = 13列
（如果有3个外生因子 → 3×4=12 → 17列）
```

### Q3: 残差连接是什么意思？

**A**: 类比
- v6.0：让模型猜"明天水温是20.5°C"（绝对值）
- v7.0：让模型猜"明天水温会上升0.3°C"（增量）
- 输出 = 今天的值(20.2) + 增量(0.3) = 20.5

### Q4: Seasonal Naive 是什么？

**A**: `y[t] = y[t-7]`
- "明天的值 = 上周同一天的值"
- 对有周模式的数据（如工作日vs周末）很强

### Q5: 如果v7.0还是不理想怎么办？

**A**:
1. 试用 **LightGBM**（用相同特征）
2. 回到 **横截面分类**（R² > 99%）
3. 接受现实：该数据不适合时序预测

---

## 📚 相关文档

- **详细说明**: `README_V7_ENHANCEMENTS.md`
- **v6修复**: `README_V6_CRITICAL_FIX.md`
- **v5发现**: `README_V5_IMPORTANT_FINDINGS.md`

---

## 🎓 总结

**v7.0 = v6.0 + 5大增强**

| 增强 | 难度 | 收益 |
|------|------|------|
| 外生因子lag/ma7 | 低 | ⭐⭐⭐ |
| 残差连接 | 中 | ⭐⭐ |
| Seasonal Naive | 低 | ⭐ |
| ReduceLROnPlateau | 低 | ⭐⭐ |
| AMP + RobustScaler | 低 | ⭐⭐ |

**推荐策略**：
```
如果 v6.0 R² > 0.6:
    → 试试v7.0，可能提升到0.65+

如果 v6.0 R² = 0.4-0.6:
    → v7.0 可能有明显提升（0.05-0.10）

如果 v6.0 R² < 0.4:
    → 时序方法不适合，改用LightGBM/横截面
```

---

**现在就运行试试吧！** 🚀

```bash
python debug_beijing_v7_enhanced.py
```

**创建日期**: 2025-10-31
**版本**: v7.0 Enhanced
**估计运行时间**: 10-30分钟
