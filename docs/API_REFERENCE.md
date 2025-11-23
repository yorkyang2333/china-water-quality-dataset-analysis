# API 参考文档

本文档记录了水质污染数据分析平台的 REST API 端点。

## 服务器位置

API 服务器代码位于：`platform_demo/backend/`

主要文件：
- `platform_demo/backend/app.py` - Flask 应用主文件
- `platform_demo/backend/models/` - 模型实现
- `platform_demo/backend/data/` - 数据目录

## 认证

所有 API 端点都需要登录认证（`@login_required` 装饰器）。

## API 端点

### 混合模型 API（v2.5 新增）

#### 1. 训练混合模型
```
POST /api/model/hybrid/train
```

**功能**：训练 LSTM + RandomForest 混合模型

**请求参数**：
- `indicator` (必需): 目标指标名称（如 "COD_mg_L"）
- `province` (可选): 省份名称（如 "Beijing"）
- `station` (可选): 监测站名称

**响应**：
```json
{
  "status": "success",
  "model_id": "lstm_rf_hybrid_20231122_143022",
  "metrics": {
    "mse": 0.123,
    "mae": 0.234,
    "r2": 0.876
  }
}
```

#### 2. 超参数优化
```
POST /api/model/hybrid/optimize
```

**功能**：使用 Optuna 进行超参数优化

**请求参数**：
- `indicator` (必需): 目标指标名称
- `province` (可选): 省份名称
- `station` (可选): 监测站名称
- `n_trials` (可选): Optuna 试验次数（默认：10）

**支持的超参数**（共11个）：
- LSTM 参数：`hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size`, `seq_length`
- RF 参数：`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- 融合权重：`fusion_weight`

**响应**：
```json
{
  "status": "success",
  "best_params": {
    "hidden_size": 128,
    "num_layers": 2,
    "n_estimators": 200,
    ...
  },
  "best_value": 0.123,
  "trials_completed": 10
}
```

#### 3. 模型预测
```
POST /api/model/hybrid/predict
```

**功能**：使用训练好的模型进行预测

**请求参数**：
- `model_id` (必需): 模型ID
- `input_data` (必需): 输入数据（JSON数组）

**响应**：
```json
{
  "status": "success",
  "predictions": [23.5, 24.1, 25.3, ...],
  "model_id": "lstm_rf_hybrid_20231122_143022"
}
```

#### 4. 模型状态
```
GET /api/model/hybrid/status?model_id=<model_id>
```

**功能**：获取模型训练状态和性能指标

**响应**：
```json
{
  "status": "trained",
  "model_id": "lstm_rf_hybrid_20231122_143022",
  "indicator": "COD_mg_L",
  "province": "Beijing",
  "metrics": {
    "mse": 0.123,
    "mae": 0.234,
    "r2": 0.876
  },
  "created_at": "2023-11-22 14:30:22"
}
```

## LSTM + RF 混合模型相关文件

| 文件 | 位置 | 功能 |
|------|------|------|
| `lstm_rf_hybrid.py` | `platform_demo/backend/models/` | 核心模型类（用于平台API） |
| `lstm_rf_hybrid_province.py` | `models/timeseries_models_province_revised/` | ⭐⭐⭐ 省份级模型 + Optuna 调参（主要使用） |
| ~~`lstm_rf_hybrid_station.py`~~ | ~~`models/archived/`~~ | ❌ 已归档，不再维护 |
| `test_hybrid_api.py` | `platform_demo/` | 平台 API 测试脚本 |

## 错误处理

所有 API 端点都包含完整的错误处理和日志记录。

**错误响应格式**：
```json
{
  "status": "error",
  "message": "错误描述",
  "code": "ERROR_CODE"
}
```

## 使用示例

### Python 示例

```python
import requests

# 训练模型
response = requests.post(
    "http://localhost:5000/api/model/hybrid/train",
    json={
        "indicator": "COD_mg_L",
        "province": "Beijing"
    },
    headers={"Authorization": "Bearer <token>"}
)

model_id = response.json()["model_id"]

# 预测
response = requests.post(
    "http://localhost:5000/api/model/hybrid/predict",
    json={
        "model_id": model_id,
        "input_data": [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]]
    },
    headers={"Authorization": "Bearer <token>"}
)

predictions = response.json()["predictions"]
```

### cURL 示例

```bash
# 训练模型
curl -X POST http://localhost:5000/api/model/hybrid/train \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"indicator": "COD_mg_L", "province": "Beijing"}'

# 超参数优化
curl -X POST http://localhost:5000/api/model/hybrid/optimize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"indicator": "COD_mg_L", "province": "Beijing", "n_trials": 20}'
```

## 测试

运行 API 测试脚本：

```bash
cd platform_demo
python test_hybrid_api.py
```

---

**创建日期**: 2025-11-22
**维护者**: Claude Code
**版本**: 1.0
