# 数据完整性检查清单（Data Integrity Checklist）

## 📋 概述

本清单用于验证污染等级分类模型的数据处理是否正确，防止数据泄露和其他数据科学错误。

**重要：任何模型在生成性能声称之前，都必须通过此清单的检查。**

---

## ✅ 数据加载与清理

- [ ] **清单 1.1**: 数据文件存在且可访问
  - 验证方法：`os.path.exists(data_path)`
  - 预期：文件存在
  - 相关文件：`core/safe_data_loader.py` - `load_raw_data()`

- [ ] **清单 1.2**: 数据文件编码正确
  - 验证方法：使用 UTF-8 编码读取 CSV
  - 预期：无编码错误
  - 命令：`pd.read_csv(path, encoding='utf-8')`

- [ ] **清单 1.3**: 原始数据中无破损行
  - 验证方法：检查数据完整性，无部分缺失的列
  - 预期：所有行完整
  - 命令：`df.info()`, `df.head()`

- [ ] **清单 1.4**: 移除重复行
  - 验证方法：`df.drop_duplicates()`
  - 预期：完全重复的行被移除
  - 相关：`core/safe_data_loader.py` - `basic_cleanup()`

---

## ✅ 数据分割（关键步骤）

- [ ] **清单 2.1**: 数据分割优先级
  - ⚠️ **CRITICAL**: 所有后续操作必须基于分割后的数据！
  - 正确顺序：原始数据 → **分割** → 统计 → 平衡 → 处理
  - 验证：检查分割代码在其他步骤之前
  - 相关：`core/safe_data_loader.py` - `split_data()`

- [ ] **清单 2.2**: Train/Test分割使用stratify
  - 验证方法：`train_test_split(..., stratify=y)`
  - 预期：各类别在Train/Test中的比例保持一致
  - 命令：比较 `y_train.value_counts(normalize=True)` 和 `y_test.value_counts(normalize=True)`

- [ ] **清单 2.3**: Train/Test无重叠
  - 验证方法：检查索引无重叠
  - 预期：len(train_indices ∩ test_indices) = 0
  - 相关：`core/leak_detector.py` - `IntegrityChecker.check_shape()`

- [ ] **清单 2.4**: 分割比例合理
  - 验证方法：检查 test_size 在 15%-30%
  - 预期：test_size 通常为 0.2 (20%)
  - 相关：`core/leak_detector.py` - `IntegrityChecker.check_split_ratio()`

- [ ] **清单 2.5**: 所有类别都出现在Train和Test中
  - 验证方法：`set(y_train) == set(y_test)`
  - 预期：Train和Test包含相同的类别
  - 命令：
    ```python
    train_classes = set(y_train)
    test_classes = set(y_test)
    assert train_classes == test_classes
    ```

---

## ✅ 统计量计算

- [ ] **清单 3.1**: 统计量仅基于Train数据
  - ⚠️ **关键**: 永远不要使用全数据或Test数据计算 mean/std
  - 验证：检查所有 `fit()` 操作只作用于 X_train
  - 错误示例：`scaler.fit(X)` 或 `scaler.fit(X_train + X_test)`
  - 正确示例：`scaler.fit(X_train)`

- [ ] **清单 3.2**: 缺失值填补使用Train统计量
  - 验证方法：检查缺失值填补的 median/mean 来自 Train
  - 预期：Test 缺失值使用 Train 的统计量填补
  - 相关：`core/safe_data_loader.py` - `handle_missing_and_inf()`

- [ ] **清单 3.3**: 异常值处理使用Train范围
  - 验证方法：检查异常值阈值来自 Train 的 Q1, Q3
  - 预期：Test 中的异常值基于 Train 定义的范围处理
  - 相关：`core/safe_data_loader.py` - `compute_train_statistics()`

---

## ✅ 特征标准化

- [ ] **清单 4.1**: Scaler仅在Train上拟合
  - 验证方法：`scaler.fit(X_train)` 而非 `scaler.fit(X)`
  - 预期：Scaler 参数来自 Train 数据
  - 错误示例：
    ```python
    scaler = StandardScaler()
    scaler.fit(X)  # ❌ 泄露！
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
  - 正确示例：
    ```python
    scaler = StandardScaler()
    scaler.fit(X_train)  # ✅ 正确
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 使用Train参数
    ```
  - 相关：`core/safe_data_loader.py` - `standardize()`

- [ ] **清单 4.2**: Scaler参数一致应用
  - 验证方法：Train 和 Test 使用相同的 scaler 对象
  - 预期：scaler.mean_ 和 scaler.scale_ 相同
  - 命令：
    ```python
    assert np.allclose(scaler.mean_, scaler_train.mean_)
    assert np.allclose(scaler.scale_, scaler_train.scale_)
    ```

---

## ✅ 类别不平衡处理

- [ ] **清单 5.1**: SMOTE/ADASYN仅应用于Train
  - ⚠️ **CRITICAL**: 绝不在Test上应用SMOTE！
  - 验证：检查 `smote.fit_resample(X_train, y_train)`
  - 错误示例：
    ```python
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)  # ❌ 泄露！
    ```
  - 正确示例：
    ```python
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)  # ✅ 正确
    # X_test 不处理
    ```
  - 相关：`core/safe_data_loader.py` - `apply_smote()`

- [ ] **清单 5.2**: SMOTE后数据量验证
  - 验证方法：检查 `len(X_train_balanced) >= len(X_train)`
  - 预期：平衡后 Train 数据增加，Test 不变
  - 相关：检查核心代码中的日志输出

- [ ] **清单 5.3**: 类别权重与数据分割一致
  - 验证方法：计算类别权重时只使用 y_train
  - 预期：weights = compute_class_weight('balanced', classes, y_train)
  - 错误示例：`compute_class_weight(..., y=y)` 全数据
  - 正确示例：`compute_class_weight(..., y=y_train)` 仅Train

---

## ✅ 模型训练

- [ ] **清单 6.1**: 模型仅在Train上训练
  - 验证方法：`model.fit(X_train, y_train)`
  - 预期：Train 数据用于训练，Test 仅用于评估
  - 相关：`core/base_classifier.py` - `train()` 方法

- [ ] **清单 6.2**: 验证集不泄露Test信息
  - 验证方法：如果使用验证集，应从 Train 中分割
  - 预期：Val ⊂ Train（验证集来自训练集）
  - 正确做法：
    ```python
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
    ```

- [ ] **清单 6.3**: 超参数调优不使用Test数据
  - 验证方法：检查 GridSearch/RandomSearch 仅在 Train/Val 上进行
  - 预期：Test 数据仅用于最终评估，不参与调参
  - 相关：避免在 X_test/y_test 上进行交叉验证

---

## ✅ 模型评估

- [ ] **清单 7.1**: 评估仅在Test上进行
  - 验证方法：`y_pred = model.predict(X_test)`
  - 预期：性能指标基于 Test 数据
  - 相关：`core/evaluator.py` - `evaluate()` 方法

- [ ] **清单 7.2**: 性能指标计算正确
  - 验证方法：检查每个指标的公式
  - 预期：
    - accuracy = 正确预测数 / 总数
    - precision = TP / (TP + FP)
    - recall = TP / (TP + FN)
    - F1 = 2 * (precision * recall) / (precision + recall)
  - 相关：`core/evaluator.py` - `evaluate()` 方法使用 sklearn 标准实现

- [ ] **清单 7.3**: 混淆矩阵来自Test预测
  - 验证方法：`confusion_matrix(y_test, y_pred_test)`
  - 预期：矩阵反映 Test 集上的分类结果
  - 错误示例：`confusion_matrix(y_train, y_pred_train)` 训练集混淆矩阵用于评估
  - 相关：`core/evaluator.py` - `plot_confusion_matrix()`

---

## ✅ 特征和目标检查

- [ ] **清单 8.1**: 不存在特征包含目标信息
  - 验证方法：检查特征名和计算方式
  - 预期：没有特征直接来自目标变量（如 `log_target_var`）
  - 相关：`core/leak_detector.py` - `check_feature_names()`

- [ ] **清单 8.2**: 特征与目标相关性检查
  - 验证方法：计算 Spearman 相关系数
  - 预期：大部分特征与目标的相关系数 < 0.7
  - 如果某个特征 r > 0.7：
    - ✅ 可能是合理的强相关特征
    - ❌ 也可能是数据泄露的迹象，需要人工审查
  - 相关：`core/leak_detector.py` - `check_correlation_anomaly()`

- [ ] **清单 8.3**: 特征中无缺失的目标值
  - 验证方法：检查特征计算公式中不含目标变量
  - 预期：所有特征独立于目标变量计算
  - 例外：某些特征可能基于同期其他指标（需明确标注）

---

## ✅ 数据分布检查

- [ ] **清单 9.1**: Train和Test分布不同
  - 验证方法：比较 Train/Test 的 mean, std
  - 预期：存在合理的分布差异，但不应完全相同
  - 检查命令：
    ```python
    X_train.mean() != X_test.mean()  # 应该不相等
    X_train.std() != X_test.std()    # 应该不相等
    ```
  - 相关：`core/leak_detector.py` - `check_data_distribution()`

- [ ] **清单 9.2**: 无数据类型错误
  - 验证方法：检查特征数据类型
  - 预期：数值特征为 float/int，非数值特征已编码
  - 命令：`X_train.dtypes`

- [ ] **清单 9.3**: 无NaN和Inf值
  - 验证方法：`X_train.isnull().sum() == 0` 且 `np.isfinite(X_train).all()`
  - 预期：所有值有限且已定义
  - 相关：`core/leak_detector.py` - `IntegrityChecker.check_no_nan()`

---

## ✅ 审计和可重复性

- [ ] **清单 10.1**: 随机种子已设置
  - 验证方法：检查 `random_state=42` 或其他固定值
  - 预期：所有随机操作使用相同的种子
  - 位置：
    - `train_test_split(..., random_state=42)`
    - `SMOTE(..., random_state=42)`
    - `np.random.seed(42)`

- [ ] **清单 10.2**: 审计日志已生成
  - 验证方法：检查是否输出日志文件
  - 预期：存在 `audit_log.json` 记录每个数据处理步骤
  - 相关：`core/safe_data_loader.py` - `save_audit_log()`

- [ ] **清单 10.3**: 结果可重复
  - 验证方法：运行两次，检查结果是否相同
  - 预期：同样的输入产生完全相同的输出
  - 相关：检查日志中是否所有步骤都文档化

---

## ✅ 数据泄露检查（综合）

| 泄露场景 | 风险等级 | 检查方式 | 预防方式 |
|---------|---------|--------|--------|
| 全数据分割前进行 SMOTE | 🔴 CRITICAL | 检查 SMOTE 发生在 split 之后 | 使用 `safe_data_loader.py` |
| 用全数据计算统计量 | 🔴 CRITICAL | 检查 fit/mean 仅用 X_train | 使用 `safe_data_loader.py` |
| Test 数据进行超参数调优 | 🔴 CRITICAL | 检查 GridSearch 仅在 Train/Val | 使用独立的验证集 |
| 特征包含目标信息 | 🟠 MAJOR | 检查相关系数和特征名称 | `leak_detector.check_feature_names()` |
| Train/Test 无正确分割 | 🟠 MAJOR | 检查索引无重叠 | 使用 `train_test_split` 并验证 |
| 缺失值填补用全数据 | 🟡 MINOR | 检查填补策略 | 使用 Train 的统计量 |
| 不同的 scaler 参数应用 | 🟡 MINOR | 检查 scaler 对象一致 | 同一 scaler 对象变换 X_train/X_test |

---

## 📋 检查流程

### 对于新模型

1. **快速检查**（开发阶段）
   - [ ] 清单 2.1-2.5（分割）
   - [ ] 清单 3.1（统计量）
   - [ ] 清单 4.1（Scaler）
   - [ ] 清单 5.1（SMOTE）
   - [ ] 清单 7.1-7.3（评估）

2. **完整检查**（发布前）
   - [ ] 全部清单项 1-10
   - [ ] 运行 `core/leak_detector.py` 自动检查
   - [ ] 人工审查代码和结果

### 自动检查脚本

```python
from core.leak_detector import DataLeakDetector, IntegrityChecker

# 数据泄露检测
leak_detector = DataLeakDetector()
leak_detector.check_correlation_anomaly(X_test, y_test)
leak_detector.check_feature_names(X_train)
leak_detector.check_data_distribution(X_train, X_test)
leak_detector.check_class_distribution(y_train, y_test)
report = leak_detector.generate_report()

# 数据完整性检查
integrity_checker = IntegrityChecker()
integrity_checker.check_no_nan(X_train, 'Train')
integrity_checker.check_no_nan(X_test, 'Test')
integrity_checker.check_shape(X_train, X_test)
integrity_checker.check_split_ratio(len(X_train), len(X_test))
integrity_checker.print_summary()
```

---

## 🚨 常见错误

### 错误1：全数据SMOTE
```python
# ❌ 错误
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)  # 全数据！
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced)

# ✅ 正确
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### 错误2：全数据计算统计量
```python
# ❌ 错误
scaler = StandardScaler()
scaler.fit(X)  # 使用全数据的mean和std
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 正确
scaler = StandardScaler()
scaler.fit(X_train)  # 仅使用Train的mean和std
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 错误3：Test数据进行超参数调优
```python
# ❌ 错误
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train + X_test, y_train + y_test)  # Test参与调优！

# ✅ 正确
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)  # 仅Train参与
y_pred = grid.predict(X_test)  # Test仅用于评估
```

---

## ✅ 签字确认

当一个模型通过此清单的所有检查后，请填写以下确认信息：

| 项目 | 值 |
|------|-----|
| 模型名称 | _____________ |
| 检查者 | _____________ |
| 检查日期 | _____________ |
| 通过项目数 | ___/75 |
| 是否允许发布 | [ ] 是 [ ] 否 |
| 备注 | _____________ |

---

## 📚 参考文献

- Kaggle: [How to Avoid Data Leakage](https://www.kaggle.com/docs/competitions#avoiding-overfitting)
- MLOps: [Data Validation Best Practices](https://mlops.community/)
- scikit-learn: [Pipeline and FeatureUnion](https://scikit-learn.org/stable/modules/compose.html)

---

**最后更新**: 2025-11-15
