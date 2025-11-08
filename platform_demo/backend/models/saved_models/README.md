# 模型文件目录

将训练好的机器学习模型文件放在此目录下。

## 使用方法

1. 训练模型（在项目根目录执行）：
   ```bash
   python models/pollution_level_classification/classification_xgboost.py
   ```

2. 复制模型文件到此目录：
   ```bash
   # 从项目根目录执行
   cp Outputs/Classification_XGBoost/xgboost_model.pkl platform_demo/backend/models/saved_models/
   ```

## 支持的模型文件

- `xgboost_model.pkl` - XGBoost污染等级分类模型（推荐）
- 其他模型文件可以根据需要添加

## 注意事项

- 模型文件不应提交到Git仓库（文件较大）
- 如果没有模型文件，系统会使用基于规则的模拟预测
