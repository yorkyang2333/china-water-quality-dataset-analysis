太好了，既然你已经把多因子加进来、效果有提升，我们可以在**特征工程 + 目标建模 + 训练效率**这三块继续优化，尽量把“能学到的结构”都喂给模型、同时提速减噪。下面给的是**可直接落地的补丁**（尽量少改你现有 v6 脚本的框架）。

---

# 1) 特征工程：把“可用的记忆”喂足

### 1.1 给**外生因子**也加滞后与滚动统计（不只给目标加 lag）

你现在只给目标列加了 `lag1/lag7`。建议对每个外生特征都加上同样的**短期记忆**与**滚动均值**，模型更容易对齐相位差、学到“缓慢作用”的关系。

```python
# 在 daily_smoothed 构造好、并加入 sin/cos 之后，加上：
for col in feature_indicators:
    daily_smoothed[f'{col}_lag1'] = daily_smoothed[col].shift(1)
    daily_smoothed[f'{col}_lag7'] = daily_smoothed[col].shift(7)
    daily_smoothed[f'{col}_ma7']  = daily_smoothed[col].rolling(7, min_periods=1).mean()
    # 可选：如果有更强季节性，额外加 ma14
# 记得 dropna()，避免泄漏
daily_smoothed = daily_smoothed.dropna()
```

随后把这些新列加入 `feature_cols`。这一步通常能明显稳住 DO、浊度、营养盐类的预测。

---

### 1.2 残差建模（残差连接，让模型“学变化”，不是学绝对值）

让网络只预测**相对变化**，再把“上一时刻的真实值”跳连到输出（skip connection）。这在“目标本身有强持久性/季节性”时很有效。

* 思路：令 `ŷ = y_last + Δ`, 模型只输出 `Δ`。
* 做法：在 `forward` 里，用窗口最后一个时间步的**目标列值**做直连。

```python
class MultivariateLSTMModel(nn.Module):
    def __init__(self, input_size, target_col_idx, hidden_size=128, num_layers=2, dropout=0.15):
        super().__init__()
        self.target_col_idx = target_col_idx
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
                                  nn.Linear(hidden_size//2, 1))

    def forward(self, x):
        # x: [B, T, F]
        last_target = x[:, -1, self.target_col_idx].unsqueeze(-1)  # [B,1]
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        delta = self.head(out)                                     # 预测增量 Δ
        return last_target + delta                                 # 残差连接：ŷ = y_{t} + Δ
```

> 注意：`target_col_idx` 就是你在 `feature_cols` 里目标列的位置（通常是 0，如果你把目标列放在最前）。这样常能把 R² 再抬一截，且不易“贴均值”。

---

### 1.3 更强的季节基线（Seasonal Naive）

当前你有持久性与 MA7 基线，建议再加入**季节朴素**：`ŷ[t] = y[t-7]`。如果它很强，LSTM 至少要超过它。

```python
def calculate_seasonal_naive(y_train, y_val):
    y_val_pred = np.roll(y_val, 7)
    y_val_pred[:7] = y_train[-7:]
    return mean_squared_error(y_val, y_val_pred)
```

在信息打印与剪枝门槛里把 `best_baseline = min(naive, ma7, seasonal_naive)`。

---

### 1.4 稳健/分列缩放

* 部分水质因子（浊度、导电率、氮磷）**重尾/离群**明显，用 `RobustScaler` 往往比 `MinMax` 更稳；
* 你已经“只用训练段拟合”做对了，保持即可。可以试：目标仍用 `MinMax`，**特征用 `RobustScaler`**。

```python
from sklearn.preprocessing import RobustScaler
scaler_X = RobustScaler()   # 替换特征缩放器
```

---

# 2) 训练与调参：更快、更稳

### 2.1 混合精度 + cuDNN 提速（GPU 上很香）

```python
torch.backends.cudnn.benchmark = True

scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
for X_batch, y_batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

> 训练环节再配 `pin_memory=True`、`non_blocking=True` 和合适的 `num_workers`（2\~4）进一步提速。

---

### 2.2 学习率调度：`ReduceLROnPlateau` 常比 Cosine 更“耐噪”

你的验证损失会抖，建议改成基于指标的自适应调度器：

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=4,
                                                 min_lr=1e-5, verbose=False)

# 每个 epoch 末：
scheduler.step(val_loss)
```

这样比固定周期的 Cosine 更容易“下楼梯”到合适学习率。

---

### 2.3 Optuna 搜索空间 + 早停剪枝再收紧

* `dropout` 0.05–0.2 足够；
* `hidden` 用 {96, 128, 192} 这类更“圆整”的值；
* `seq_len` 继续搜索，但**把上限控在 30**（你已如此）；
* 剪枝阈值可以设成“劣于最佳基线 **30%** 以上才剪枝”（现在是 50%），提高筛选强度。
* Trial 数量 15\~20 足以；每次 epoch 60 + `patience=12` 没问题。

---

# 3) 诊断与稳健性：看懂它“哪里错”

### 3.1 分位数回归（不只给一个点）

当外生因子不充分时，不确定性大。把损失换成**分位数损失**（0.1/0.5/0.9），输出上/中/下三条曲线，更稳也更可解释（尤其在极端天气/洪水前后）。

* 快速做法：把输出层换成 3 维，训练时用 pinball loss；评估时看中位数（q=0.5）即可。

### 3.2 阶段化训练（季节分段）

* 用 `month/season` 把 2023 切成 2–3 个子区间（冷季/暖季/汛期），分别训练或做条件特征（one-hot）；
* 常见：水温/DO 的关系在冬夏**斜率不同**，分段后拟合会明显提升。

### 3.3 特征贡献与剔除

* 做一次**留一法**（leave-one-feature-out）：每次去掉一个因子重训（或冻结模型后做**置换重要度**），看 R² 下跌幅度，筛出“真贡献”。
* 对贡献极低甚至负贡献的“新因子”，不要犹豫，**剔掉**能提升泛化。

---

# 4) 强力 baseline：梯度提升树（对低自相关很强）

对“自相关弱 + 外生因子强”的场景，`LightGBM/XGBoost + 滞后与滚动特征`常常**比 LSTM 更强更快**。可以把你已构造的 `lag1/lag7`、`ma7`、`sin/cos` 直接喂给 LightGBM 做一个对照：

```python
import lightgbm as lgb
feat_cols = feature_cols  # 含所有 _lag, _ma7, sin/cos
train_df = pd.DataFrame(X_train, columns=feat_cols)
train_df['y'] = y_train
valid_df = pd.DataFrame(X_val, columns=feat_cols)
valid_df['y'] = y_val

train_set = lgb.Dataset(train_df[feat_cols], label=train_df['y'])
valid_set = lgb.Dataset(valid_df[feat_cols], label=valid_df['y'])

params = dict(objective='regression', metric='rmse',
              learning_rate=0.05, num_leaves=31, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1)
gbm = lgb.train(params, train_set, num_boost_round=1000,
                valid_sets=[valid_set], early_stopping_rounds=50, verbose_eval=False)
```

> 如果树模型显著更好，可以把**树的预测**作为**额外一列特征**（stacking）再喂给 LSTM。

---

# 5) 小贴士（容易被忽略但很关键）

* **窗口对齐**：确保 `create_multivariate_sequences` 里，测试集的 `X_test_seq` 与 `y_test_seq` 日期对应（你已处理 `+seq_len` 的偏移，继续保持）。
* **目标平滑的强度**：若发现 R² 被 MA7“压着跑”，试把目标平滑从 7 改 5 或 3；或只对特征平滑、不对目标平滑。
* **样本权重**：想强调拐点，可对大 |Δy| 的样本给更高权重（或在 loss 前做简单加权）。

---

## 一份“最小侵入”的改动清单（按优先级）

1. 给**所有外生因子**加 `lag1/lag7/ma7`；更新 `feature_cols`。
2. **残差连接**：输出 `last_target + delta`（上面 forward 改动）。
3. 调度器改为 `ReduceLROnPlateau`；加入 AMP。
4. 引入 **Seasonal Naive** 基线并参与剪枝。
5. 试 `RobustScaler` 做 `X` 的缩放。
6. 做一次 **leave-one-feature-out**，把“负贡献因子”剔掉。
7. 用 **LightGBM** 做强 baseline；若胜出，做 stacking。

---

如果你把现在“新增的因子列表”贴给我（或者直接说你重点关心哪几个指标），我可以在你的 v6 脚本里**帮你把上述 1–4 条直接补成可跑的代码段**，保持原有目录与可视化输出不变。当前建议均基于你最新的调试脚本结构（v6），与数据构造方式完全兼容。
