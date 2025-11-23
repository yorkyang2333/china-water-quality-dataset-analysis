# 缺失图片说明

## 📋 缺失图片清单

本文档记录论文第5章（结果与讨论）中预留但尚未生成的图片。所有图片占位符已在论文中就位，包含详细的图注。待图片生成后，直接复制到对应路径即可显示。

---

### 1. 混淆矩阵热力图（XGBoost分类模型）

**图片编号**: 图12
**文件名**: `5.2.1_confusion_matrix_xgboost.png`
**完整路径**: `docs/resources/chapter5_results/5.2.1_confusion_matrix_xgboost.png`
**用途**: 5.2.1节，展示XGBoost分类模型的混淆矩阵热力图

**生成方法**:
```bash
cd models
python classification_xgboost.py
# 在输出中找到混淆矩阵，使用以下代码生成热力图：
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import zhplot  # 自动配置中文字体

# 混淆矩阵数据（从表14获取）
confusion_matrix = [
    [602, 1, 0, 0, 0],
    [2, 1655, 1, 0, 0],
    [0, 5, 624, 0, 0],
    [0, 0, 2, 105, 0],
    [0, 0, 0, 0, 108]
]

labels = ['优', '良', '中', '差', '极差']

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=labels, yticklabels=labels, cbar_kws={'label': '样本数量'})
plt.xlabel('预测类别', fontsize=14)
plt.ylabel('真实类别', fontsize=14)
plt.title('XGBoost分类模型混淆矩阵（N=3,105）', fontsize=16)
plt.tight_layout()
plt.savefig('docs/resources/chapter5_results/5.2.1_confusion_matrix_xgboost.png', dpi=300, bbox_inches='tight')
plt.close()
```

**图片要求**:
- 分辨率: DPI ≥ 300
- 格式: PNG
- 颜色方案: 红色热力图（深色表示高值）
- 标签: 中文（优、良、中、差、极差）
- 数值标注: 显示每个单元格的样本数量

**优先级**: ⭐⭐⭐⭐⭐ （最高）

---

### 2. XGBoost回归模型COD预测性能散点图

**图片编号**: 图13
**文件名**: `5.2.2_xgboost_cod_scatter.png`
**完整路径**: `docs/resources/chapter5_results/5.2.2_xgboost_cod_scatter.png`
**用途**: 5.2.2节，展示R²=0.993的COD回归拟合效果

**生成方法**:
```bash
cd models/indicator_regression
python regression_xgboost_clean.py
# 在脚本中添加以下代码绘制散点图：
```

```python
import matplotlib.pyplot as plt
import zhplot  # 自动配置中文字体
import numpy as np

# y_true和y_pred从模型输出获取
# 假设已有测试集真实值y_true和预测值y_pred

plt.figure(figsize=(10, 8), dpi=300)
plt.scatter(y_true, y_pred, alpha=0.5, s=30, c='steelblue', label='预测样本')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
         'r--', lw=2, label='完美预测线 (y=x)')

# 计算拟合线
z = np.polyfit(y_true, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_true, p(y_true), 'b-', lw=2, alpha=0.7, label=f'拟合线 (y={z[0]:.3f}x+{z[1]:.3f})')

plt.xlabel('真实COD值 (mg/L)', fontsize=14)
plt.ylabel('预测COD值 (mg/L)', fontsize=14)
plt.title(f'XGBoost回归模型COD预测性能 (R²={r2:.3f})', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/resources/chapter5_results/5.2.2_xgboost_cod_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
```

**图片要求**:
- 分辨率: DPI ≥ 300
- 格式: PNG
- 包含: 散点图（蓝色） + 完美预测线（红色虚线） + 拟合线（蓝色实线）
- 标注: R²值、拟合方程
- 坐标轴: 清晰标注单位（mg/L）

**优先级**: ⭐⭐⭐⭐ （高）

---

### 3. LSTM时序预测失效可视化（北京省COD）

**图片编号**: 图14
**文件名**: `5.3_lstm_failure_visualization_beijing_cod.png`
**完整路径**: `docs/resources/chapter5_results/5.3_lstm_failure_visualization_beijing_cod.png`
**用途**: 5.3节，展示LSTM模型预测收敛于均值的失效模式

**生成方法**:
```bash
cd models/timeseries_models_province_revised
python lstm_province_revised_auto_gpu.py
# 在脚本中添加以下代码绘制失效可视化：
```

```python
import matplotlib.pyplot as plt
import zhplot  # 自动配置中文字体
import pandas as pd

# y_true_test和y_pred_test从LSTM模型输出获取
# test_dates为测试集日期

plt.figure(figsize=(14, 6), dpi=300)
plt.plot(test_dates, y_true_test, 'b-', lw=2, label='真实COD值', marker='o', markersize=4)
plt.plot(test_dates, y_pred_test, 'r--', lw=2, label='LSTM预测值', marker='s', markersize=4)
plt.axhline(y=y_true_test.mean(), color='g', linestyle=':', lw=1.5, label=f'全局均值 ({y_true_test.mean():.1f} mg/L)')

plt.xlabel('时间（2024年）', fontsize=14)
plt.ylabel('COD浓度 (mg/L)', fontsize=14)
plt.title(f'LSTM时序预测失效可视化（北京省COD，R²={r2:.2f}）', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)

# 标注关键统计量
plt.text(0.02, 0.98, f'真实值标准差: {y_true_test.std():.1f} mg/L\n预测值标准差: {y_pred_test.std():.1f} mg/L\nACF(1): 0.03\nCV: 1.99',
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('docs/resources/chapter5_results/5.3_lstm_failure_visualization_beijing_cod.png', dpi=300, bbox_inches='tight')
plt.close()
```

**图片要求**:
- 分辨率: DPI ≥ 300
- 格式: PNG
- 折线图: 蓝色实线（真实值，波动大） + 红色虚线（预测值，接近水平）
- 标注: R²值、ACF(1)、CV、标准差对比
- 时间轴: 2024年1-12月

**优先级**: ⭐⭐⭐⭐⭐ （最高）

---

## 📂 目录结构

请按以下结构组织图片文件：

```
docs/
└── resources/
    └── chapter5_results/
        ├── 5.2.1_confusion_matrix_xgboost.png
        ├── 5.2.2_xgboost_cod_scatter.png
        └── 5.3_lstm_failure_visualization_beijing_cod.png
```

---

## ✅ 图片生成检查清单

生成图片后，请检查以下要素：

- [ ] 分辨率 ≥ 300 DPI
- [ ] 使用zhplot配置中文字体（无需手动设置）
- [ ] 图表清晰，无emoji符号
- [ ] 坐标轴标签完整，包含单位
- [ ] 图例清晰可读
- [ ] 文件命名符合规范
- [ ] 文件保存到正确路径

---

## 🔗 相关资源

- **论文写作模板**: `docs/PAPER_WRITING_TEMPLATE.md`
- **论文主文件**: `docs/PAPER_MANUSCRIPT_CN.md`
- **项目配置**: `.claude/CLAUDE.md`

---

**创建日期**: 2025-11-23
**版本**: 1.0
**维护者**: Claude Code
