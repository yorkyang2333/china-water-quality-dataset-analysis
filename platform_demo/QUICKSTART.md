# 快速开始指南 | Quick Start Guide

## Windows 用户

### 方法一：使用启动脚本（最简单）

1. 双击 `start.bat` 文件
2. 等待依赖安装和服务器启动
3. 浏览器访问 http://localhost:5000

### 方法二：使用 Python 启动

```cmd
pip install -r backend\requirements.txt
python run.py
```

### 方法三：手动启动

```cmd
cd backend
pip install -r requirements.txt
python app.py
```

然后访问: http://localhost:5000

---

## Mac/Linux 用户

### 推荐方式

```bash
pip install -r backend/requirements.txt
python run.py
```

### 或手动启动

```bash
cd backend
pip install -r requirements.txt
python app.py
```

然后访问: http://localhost:5000

---

## 常见问题

### Q1: 提示"找不到模块"

**解决方法:**
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost
```

### Q2: 端口5000被占用

**解决方法:**
编辑 `backend/app.py`，修改端口号：
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # 改为5001或其他
```

同时修改 `frontend/js/map.js`：
```javascript
const API_BASE_URL = 'http://localhost:5001/api';
```

### Q3: 地图不显示

**原因:** 可能是网络问题，无法加载OpenStreetMap瓦片

**解决方法:** 检查网络连接，或更换地图源

### Q4: 预测功能返回mock数据

**原因:** 未找到训练好的模型文件

**解决方法:**
1. 确保XGBoost模型文件存在于: `../Outputs/Classification_XGBoost/xgboost_model.pkl`
2. 或者先运行 `classification_xgboost.py` 训练模型

---

## 功能测试清单

- [ ] 地图正常显示，可以看到监测站点
- [ ] 点击站点弹出详细信息
- [ ] 选择时间范围，点击"应用"
- [ ] 使用矩形/圆形工具选择区域
- [ ] 右侧查看选中区域统计
- [ ] 输入水质指标，预测污染等级
- [ ] 选择省份和日期，预测水质指标
- [ ] 左下角趋势图表正常显示

---

## 下一步

阅读完整的 [README.md](README.md) 了解更多功能和API文档。

---

祝使用愉快！🎉
