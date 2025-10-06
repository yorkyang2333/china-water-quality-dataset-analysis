# 水质污染数据分析项目 - Claude 配置

## 项目背景
这是一个水质污染数据分析项目，包含数据预处理、探索性数据分析(EDA)和机器学习预测模型。项目涵盖了时间序列预测和横截面预测两大类方法。

## 📁 项目结构

```
Codes/
├── .claude/                              # Claude配置文件
│   └── CLAUDE.md                         # 本文件
├── preprocessed_data.csv                 # 预处理后的数据（主数据源）
├── preprocess.py                         # 数据预处理脚本
├── eda.py                                # 探索性数据分析脚本
├── determine_ranges.py                   # 数据范围确定脚本
├── requirements.txt                      # Python依赖包列表
│
├── timeseries_models_province/           # ⭐ 时间序列模型 - 按省份建模
│   ├── README.md                         # 省份模型说明文档
│   ├── ar_province_prediction.py         # AR模型（CPU版）
│   ├── arima_province_prediction.py      # ARIMA模型（CPU版）
│   ├── arima_province_prediction_parallel.py  # ARIMA多进程加速版
│   ├── sarima_province_prediction.py     # SARIMA模型（CPU版）
│   ├── sarima_province_prediction_parallel.py # SARIMA多进程加速版
│   ├── lstm_province_prediction.py       # LSTM模型（PyTorch CPU版）
│   ├── lstm_province_prediction_gpu.py   # LSTM模型（GPU加速版）
│   └── prophet_province_prediction.py    # Prophet模型
│
├── timeseries_models_station/            # 时间序列模型 - 按监测站建模
│   ├── README.md                         # 监测站模型说明文档
│   ├── ar_station_prediction_*.py        # AR模型（多版本）
│   ├── arima_station_prediction_*.py     # ARIMA模型（多版本）
│   ├── sarima_station_prediction_*.py    # SARIMA模型（多版本）
│   ├── lstm_station_prediction_*.py      # LSTM模型（多版本）
│   └── prophet_station_prediction_*.py   # Prophet模型（多版本）
│
├── backup_prediction_attempts_crosssectional/  # 横截面预测尝试（备份）
│   ├── README.md                         # 横截面模型说明
│   ├── baseline_prediction.py            # 基线模型
│   ├── crosssectional_prediction.py      # 横截面预测
│   ├── improved_province_prediction.py   # 改进的省份预测
│   └── ...                               # 其他实验脚本
│
└── Outputs/                              # 所有模型的输出结果
    ├── EDA/                              # 探索性数据分析结果
    ├── AR_Province_Predictions/          # AR省份模型输出
    ├── ARIMA_Province_Predictions/       # ARIMA省份模型输出
    ├── SARIMA_Province_Predictions/      # SARIMA省份模型输出
    ├── LSTM_Province_Predictions/        # LSTM省份模型输出
    ├── LSTM_Province_Predictions_GPU/    # LSTM GPU版本输出
    └── ...                               # 其他模型输出
```

## 🎯 核心数据集特点（重要！）

### ✅ 优势
- **数据量充足**：共180个监测站，每个监测站平均86个观测点（中位数78，范围56-182）
- **覆盖范围广**：10个省份，多个城市，多个监测站
- **指标齐全**：16个水质指标（温度、pH、溶解氧、重金属等）

### ⚠️ 关键问题：时间间隔极不规则

**这是影响模型选择的最重要因素！**

| 统计指标 | 数值 | 说明 |
|---------|------|------|
| 平均间隔 | 7-10天 | 看似正常 |
| 间隔标准差 | **17-29天** | ⚠️ 变化巨大！ |
| 最小间隔 | 0天 | 同日多次采样 |
| 最大间隔 | **127-222天** | ⚠️ 超过7个月的空白期 |
| **结论** | **不规则抽样** | ❌ 不是定期时间序列监测 |

### 📊 数据特征分析
- 更像"事件驱动采样"（发现污染时才检测）
- 缺乏可预测的时间依赖结构
- 大段空白期打断了时间连续性
- 违反了时间序列模型的等间隔采样假设

## 🚫 不适合的方法（已验证）

### 时间序列模型（效果差，不推荐）

**原因**：时间间隔极不规则，违反时间序列模型基本假设

| 模型类型 | 实测效果 | 为什么失败 |
|---------|---------|-----------|
| **ARIMA/SARIMA** | R² < 0（负值） | 假设等间隔采样，无法处理不规则间隔 |
| **LSTM（深度学习）** | R² < 0（负值） | 序列长度要求高，空白期打断学习 |
| **Prophet** | R² < 0（负值） | 虽对缺失值鲁棒，但间隔过大仍失效 |
| **AR（自回归）** | R² < 0（负值） | 需要稳定的时间依赖关系 |

**结论**：预测效果比简单平均值还差！

### 时间序列模型的保留原因
虽然效果差，但我们保留了完整的时间序列模型实现：
- 📚 **教育目的**：展示模型实现和失败原因
- 🔬 **对比实验**：与其他方法对比，证明数据特性的重要性
- 📊 **文档完整性**：记录所有尝试过的方法

## ✅ 推荐方法（有效）

### 1. 横截面预测（最推荐）
- **方法**：Random Forest、XGBoost、Gradient Boosting
- **特征**：经纬度、省份、城市、监测站ID、历史统计值
- **优势**：不依赖时间序列的连续性
- **位置**：`backup_prediction_attempts_crosssectional/`

### 2. 多指标回归
- **方法**：利用其他水质指标预测目标指标
- **特征**：pH、温度、溶解氧、浊度、重金属等
- **优势**：利用指标间的相关性

### 3. 分类模型
- **目标**：预测污染等级（Excellent/Good/Moderate/Poor/Very Poor）
- **方法**：SVM、Random Forest、Neural Networks

## 📝 代码风格规范

### 注释和文档
- 使用**中文注释**和文档字符串（方便团队成员理解）
- 使用**英文变量名**和函数名（遵循Python规范）
- 每个脚本包含详细的方法说明和适用场景

### 文件命名规范
```
{model}_{level}_prediction_{version}.py

model: ar, arima, sarima, lstm, prophet, rf, xgboost
level: province（省份）, station（监测站）, cross（横截面）
version: cpu, gpu, parallel（可选）

示例：
- arima_province_prediction.py
- lstm_province_prediction_gpu.py
- arima_station_prediction_parallel.py
```

## 📊 输出规则

### 目录结构
```
Outputs/{ModelName}_{Level}_Predictions_{Version}/
├── report.txt                    # 文本报告（UTF-8编码）
├── {Province/Station}_*.png      # 可视化图表
└── predictions.csv               # 预测结果（可选）
```

### 报告内容要求
1. **方法说明**（中英文双语）
2. **性能指标**：R²、RMSE、MAE等
3. **结果排序**：按性能从好到差
4. **建议和结论**
5. **局限性说明**（特别是时间序列模型）

### 可视化要求
- 使用**中英文双语标题**
- 保存为**高分辨率PNG** (dpi=150-300)
- 使用 `matplotlib` + `seaborn`
- 支持中文字体：`'Arial Unicode MS'`, `'SimHei'`, `'DejaVu Sans'`

## 🚀 创建新模型指南

### 1. 选择合适的方法
```python
# ✅ 推荐（横截面/特征工程）
if 数据特征丰富 and 不依赖时间顺序:
    使用 Random Forest / XGBoost / Gradient Boosting

# ⚠️ 谨慎使用（时间序列）
if 用户明确要求时间序列:
    使用 ARIMA / LSTM / Prophet
    但需明确说明：效果差，R² < 0
```

### 2. 文件组织
```python
# 创建输出目录
OUTPUT_DIR = 'Outputs/{ModelName}_{Level}_Predictions_{Version}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### 3. 数据加载
```python
# 优先使用预处理数据
df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')

# 数据集包含以下列：
# - Province: 省份
# - City: 城市
# - Monitoring_Station: 监测站
# - Latitude, Longitude: 经纬度
# - Date: 日期
# - Water_Temperature_C, pH, ... : 16个水质指标
# - Water_Quality_Index: 水质指数
# - Pollution_Level: 污染等级
```

### 4. 性能评估
```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 必须包含的指标
metrics = {
    'R2': r2_score(y_true, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
    'MAE': mean_absolute_error(y_true, y_pred),
    'Sample_Size': len(y_true)
}
```

## 📖 重要参考文件

### 数据分析
- `Outputs/EDA/` - 探索性数据分析结果
  - 数据分布、相关性分析
  - 时间序列特性分析（证明时间间隔不规则）
  - 空间分布可视化

### 模型说明
- `timeseries_models_province/README.md` - 省份级时间序列模型
- `timeseries_models_station/README.md` - 监测站级时间序列模型
- `backup_prediction_attempts_crosssectional/README.md` - 横截面预测方法

## ⚠️ 常见陷阱

### 1. 时间序列模型的误用
```python
# ❌ 错误：直接使用时间序列模型
# 结果：R² < 0，效果极差
model = ARIMA(...)
model.fit(data)

# ✅ 正确：需要明确警告用户
print("⚠️ 警告：数据时间间隔不规则，时间序列模型效果差！")
print("建议：使用横截面预测或多指标回归")
```

### 2. 数据分割
```python
# ❌ 错误：随机分割（破坏时间序列）
train_test_split(X, y, random_state=42)

# ✅ 正确：按时间顺序分割
data = data.sort_values('Date')
split_idx = int(len(data) * 0.8)
train = data[:split_idx]
test = data[split_idx:]
```

### 3. 特征工程
```python
# ✅ 横截面预测可用的特征
features = [
    'Latitude', 'Longitude',           # 空间特征
    'Province', 'City',                # 分类特征
    'pH', 'Temperature', 'DO',         # 水质指标
    # ❌ 避免使用：Date（时间序列特征）
]
```

## 🔧 环境配置

### Python依赖
```bash
pip install -r requirements.txt
```

### 主要库
- **数据处理**：pandas, numpy
- **机器学习**：scikit-learn, xgboost
- **深度学习**：torch (PyTorch), tensorflow (可选)
- **时间序列**：statsmodels, prophet
- **可视化**：matplotlib, seaborn

### GPU支持（可选）
```bash
# LSTM GPU版本需要
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📞 快速开始

### 运行探索性数据分析
```bash
python3 eda.py
# 输出：Outputs/EDA/
```

### 运行时间序列模型（按省份）
```bash
# CPU版本
python3 timeseries_models_province/arima_province_prediction.py

# GPU加速版本
python3 timeseries_models_province/lstm_province_prediction_gpu.py

# 多进程并行版本
python3 timeseries_models_province/arima_province_prediction_parallel.py
```

### 运行横截面预测（推荐）
```bash
python3 backup_prediction_attempts_crosssectional/crosssectional_prediction.py
```

## 📚 总结

### 核心建议
1. ✅ **优先使用横截面预测/特征工程方法**
2. ⚠️ **避免纯时间序列方法**（除非用户明确要求）
3. 📊 **重视数据特性分析**（时间间隔、空间分布、指标相关性）
4. 📝 **详细记录实验结果**（成功和失败都很重要）

### 关键认识
- **数据特性决定方法选择**：时间间隔不规则 → 不适合时间序列
- **模型复杂度≠效果好**：简单的横截面预测可能比复杂的LSTM效果好
- **失败也是经验**：保留失败的时间序列模型，作为对比和教训

---

**创建日期**: 2025-01-05
**最后更新**: 2025-10-06
**维护者**: 团队项目
**版本**: 2.0（重构后）
