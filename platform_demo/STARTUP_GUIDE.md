# 水质监测平台 - 快速启动指南

## 🚀 一键启动（推荐）

```bash
cd platform_demo
python start_all.py
```

这个脚本会自动：
- ✅ 检查环境（Python、Node.js、npm）
- ✅ 安装Vue依赖（如果需要）
- ✅ 启动Flask后端（http://localhost:5000）
- ✅ 启动Vue前端（http://localhost:3000）
- ✅ 自动打开浏览器

**停止服务**：按 `Ctrl+C`

---

## 📋 分别启动

### 1. 启动后端
```bash
python run.py
```

### 2. 启动Vue前端
```bash
cd platform_demo
python test_vue_frontend.py
```

或手动启动：
```bash
cd platform_demo/frontend-vue
npm install  # 首次运行
npm run dev
```

---

## 🔧 已修复的问题

### 问题1: Unicode编码错误
**症状**：脚本中的emoji无法显示，导致程序崩溃
**原因**：Windows中文环境下默认使用GBK编码
**解决**：在脚本开头添加UTF-8编码支持

### 问题2: subprocess无法检测Node.js
**症状**：明明安装了Node.js，但脚本显示未安装
**原因**：Windows下subprocess需要`shell=True`参数
**解决**：所有subprocess调用添加`shell=True`

### 问题3: Vue依赖安装失败
**症状**：`vue-leaflet@^0.7.1`版本不存在
**原因**：Vue 2的包不兼容Vue 3
**解决**：改用`@vue-leaflet/vue-leaflet@^0.10.1`

### 问题4: 前端页面空白
**症状**：控制台显示图标导入错误
**原因**：`@element-plus/icons-vue`中不存在某些图标
**解决**：使用正确的图标名称：
- `Water` → `Odometer`
- `DataBoard` → `DataLine`
- `Monitor` → `View`
- `DataAnalysis` → `PieChart`

### 问题5: favicon 404错误
**解决**：创建了渐变水滴形状的favicon.svg

---

## 📡 服务地址

- **后端API**: http://localhost:5000
- **前端界面**: http://localhost:3000
- **API文档**: http://localhost:5000/api/docs

---

## 🎨 Vue.js前端特性

- ✨ Vue 3 + Composition API
- 🎨 Element Plus UI组件库
- 📊 Chart.js 图表库
- 🗺️ Vue-Leaflet 地图组件
- 🔄 Pinia 状态管理
- 🛣️ Vue Router 路由系统
- ⚡ Vite 构建工具（超快热重载）
- 📱 响应式设计

---

## 💡 开发提示

1. **修改代码会自动热重载** - 保存文件后浏览器会自动刷新
2. **前端代理** - 前端会自动将`/api/*`请求代理到后端
3. **错误调试** - 打开浏览器控制台查看详细错误信息
4. **端口冲突** - 如果端口被占用，可以在代码中修改端口号

---

**创建时间**: 2025-11-07
**维护者**: Claude Code
