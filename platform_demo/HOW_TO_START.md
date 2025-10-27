# 如何启动平台 | How to Start

## ✅ 导入错误已修复

之前遇到的 `ModuleNotFoundError: No module named 'backend'` 错误已经修复！

---

## 🚀 三种启动方式

### 方式1：使用 run.py（最简单，推荐）

```bash
cd water-quality-platform
python run.py
```

**优点：**
- ✅ 一条命令搞定
- ✅ 自动处理路径问题
- ✅ 跨平台支持

---

### 方式2：使用 start.bat（Windows用户）

1. 进入项目目录
2. 双击 `start.bat` 文件
3. 等待依赖安装（首次运行）
4. 服务器自动启动

**优点：**
- ✅ 图形化操作
- ✅ 自动安装依赖
- ✅ 错误提示友好

---

### 方式3：使用 test_server.py（带诊断信息）

```bash
cd water-quality-platform
python test_server.py
```

**优点：**
- ✅ 显示详细启动信息
- ✅ 逐步检查每个模块
- ✅ 适合调试问题

---

## 📋 完整启动流程

### 步骤1：安装依赖（首次运行）

```bash
cd water-quality-platform
pip install -r backend/requirements.txt
```

**依赖列表：**
- flask==2.3.0
- flask-cors==4.0.0
- pandas==2.0.0
- numpy==1.24.0
- scikit-learn==1.3.0
- xgboost==1.7.0
- joblib==1.3.0

### 步骤2：启动服务器

选择以下任一方式：

**方式A - 最简单：**
```bash
python run.py
```

**方式B - 带检查：**
```bash
python test_server.py
```

**方式C - 传统方式：**
```bash
cd backend
python app.py
```

### 步骤3：访问平台

打开浏览器，访问：
```
http://localhost:5000
```

---

## ✅ 启动成功的标志

你应该看到以下输出：

```
============================================================
  Water Quality Monitoring Platform - Backend Server
============================================================

  Data loaded: 15524 records
  Provinces: 10
  Stations: 180

  Server running at: http://localhost:5000
============================================================

 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in production.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
```

---

## 🔧 问题排查

### 问题1: ModuleNotFoundError

**症状：**
```
ModuleNotFoundError: No module named 'backend'
```

**解决方案：**
使用 `run.py` 或 `test_server.py` 启动，不要直接运行 `backend/app.py`

---

### 问题2: 找不到数据文件

**症状：**
```
Warning: Data file not found at ...
```

**解决方案：**
确保 `preprocessed_data_feature_eng.csv` 文件在项目根目录的上一级：
```
china-water-quality-dataset-analysis/
├── preprocessed_data_feature_eng.csv  ← 应该在这里
└── water-quality-platform/
    └── run.py
```

---

### 问题3: 端口被占用

**症状：**
```
OSError: [Errno 48] Address already in use
```

**解决方案：**
1. 关闭占用5000端口的程序
2. 或修改端口号（编辑 `run.py` 和 `frontend/js/map.js`）

---

### 问题4: 依赖安装失败

**症状：**
```
ERROR: Failed to install dependencies
```

**解决方案：**
手动安装每个包：
```bash
pip install flask
pip install flask-cors
pip install pandas
pip install numpy
pip install scikit-learn
pip install xgboost
```

---

## 📊 验证平台功能

启动成功后，检查以下功能：

1. **地图显示** - 应该看到中国地图和监测站点
2. **站点信息** - 点击站点标记，弹出详细信息
3. **控制面板** - 左侧可以选择时间范围和省份
4. **趋势图表** - 左下角显示趋势图
5. **预测功能** - 右侧可以输入数据进行预测

---

## 🎯 快速测试API

服务器启动后，可以在新的命令行窗口测试API：

```bash
# 测试获取站点
curl http://localhost:5000/api/stations

# 测试统计摘要
curl http://localhost:5000/api/statistics/summary

# 测试趋势数据
curl http://localhost:5000/api/trends?indicator=Water_Quality_Index
```

---

## 📱 访问方式

### 本机访问
```
http://localhost:5000
http://127.0.0.1:5000
```

### 局域网访问
```
http://你的IP地址:5000
例如: http://192.168.1.100:5000
```

**查看本机IP：**
- Windows: `ipconfig`
- Mac/Linux: `ifconfig` 或 `ip addr`

---

## 🛑 停止服务器

在运行服务器的终端窗口按：
```
Ctrl + C
```

---

## 💡 提示

1. **首次启动较慢**：需要加载15,524条数据记录
2. **模型预测**：如果没有训练好的模型，会使用mock预测
3. **浏览器推荐**：Chrome、Firefox、Edge 最佳
4. **网络连接**：需要网络加载地图瓦片（OpenStreetMap）

---

## 📖 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 [QUICKSTART.md](QUICKSTART.md) 常见问题
- 查看 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) 技术架构

---

**祝使用愉快！🎉**

如有问题，请检查控制台输出的错误信息。
