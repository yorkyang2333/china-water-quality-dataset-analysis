# 水质监测平台（精简开源版）

该仓库提供一个可在本地或服务器快速部署的 **水质监测数据可视化平台**。
仓库中已移除所有可能泄露隐私的原始数据与账号信息，并内置一份仅用于演示的样例数据，便于在任何环境下一键运行。

## 重要更新（2025-11-14）

### 数据泄露问题发现与修复

在XGBoost回归模型开发过程中，发现了严重的**数据泄露（Data Leakage）**问题：

**问题描述**：
- `models/indicator_regression/regression_xgboost_improved.py` 中的15个高级特征中，有**11个特征直接使用了预测目标本身或其变换**
- 例如：`log_Ammonia_N`特征与目标`Ammonia_N_mg_L`的相关系数高达**r=0.996**（几乎完美相关）
- 导致模型出现**虚假的高性能**：R²=0.9993

**真实性能**：
- 使用无泄露特征的真实性能：**R²=-0.037**（远低于Naive基线）
- 性能差距：**1.036（103倍！）**

**解决方案**：
1. 创建了无泄露版本：`models/indicator_regression/regression_xgboost_clean.py`
2. 仅使用4个干净特征（不依赖预测目标）
3. 生成完整对比报告：`docs/DATA_LEAKAGE_FINAL_REPORT.md`（22KB详细分析）

**适用场景**：
- ✅ **同时刻缺失值填补**：当传感器故障导致某指标缺失，但其他指标正常时，可使用`improved`版本（R²=0.9993）
- ❌ **未来预测**：由于未来的目标值未知，无法计算泄露特征，必须使用`clean`版本（R²=-0.037）

**警告标注**：
- `regression_xgboost_improved.py`文件头部已添加72行严重警告
- `.claude/CLAUDE.md`已更新至v2.6，新增数据泄露检测规则
- 本README已标注（当前章节）

---

### 新增EDA功能

**1. 时空分布分析** (`scripts/spatial_distribution_analysis.py`)
- 省际差异ANOVA检验：仅**浊度**显示显著差异
- K-means聚类：10省分为**3个水质类别**
- PCA降维可视化（解释**43.1%**方差）

**2. 时间分布分析** (`scripts/temporal_distribution_analysis.py`)
- **关键发现**：2024年采样密度仅为2023年的**12.4%**
- KS检验：5个指标发生显著分布漂移
- 这解释了为何时间序列方法失败（R²<0）

**3. 贝叶斯网络分析** (`scripts/bayesian_network_analysis.py`)
- 构建专家知识网络：**17条因果关系**
- 基于物理化学原理（温度→DO、氮循环、有机物降解等）
- 网络结构可视化

---

### 论文撰写建议

如果您使用本项目数据进行学术研究，请注意：

1. **必须报告无泄露版本的真实性能**（R²=-0.037），而非虚假的0.9993
2. 有泄露版本仅适用于**缺失值填补场景**，需在论文中明确说明应用场景
3. 时间序列方法失败有系统性证据（ACF=0.03, CV=1.99, 分布漂移），可作为**失败案例分析**
4. **成功的方法**：
   - 污染等级分类：准确率**99.5%**（XGBoost/RandomForest）
   - 化学污染物回归：部分指标R²可达**0.97**（横截面方法）

详细分析请参阅：
- `docs/DATA_LEAKAGE_FINAL_REPORT.md` - 数据泄露完整报告
- `models/timeseries_models_province_revised/DEPRECATED_ANALYSIS.md` - 时间序列失败分析
- `Outputs/Spatial_Distribution_Analysis/` - 空间分布分析结果
- `Outputs/Temporal_Distribution_Analysis/` - 时间分布分析结果

---

## 功能概览
- Flask + SQLite 后端：提供监测数据查询、预测接口以及用户注册、登录、资料维护等 API。
- Vue 3 + Element Plus 前端：可视化展示仪表盘、趋势分析、空间监测等多种页面，并支持个人资料管理。
- 一键启动脚本：`python platform_demo/start_all.py` 会自动检测并安装依赖，依次启动后端与前端，同时打开浏览器。

## 仓库结构
```
platform_demo/
├── backend/            # Flask 后端
│   ├── app.py          # 主入口，自动加载样例或自定义数据集
│   ├── models/         # 预测模型、用户管理等模块
│   ├── data/           # 运行时生成的 SQLite 数据库存放位置（已忽略）
│   ├── sample_data/    # 内置样例数据，支持开箱即用
│   └── requirements.txt
├── frontend-vue/       # Vue 3 前端工程
│   ├── src/            # 页面、路由、Pinia store 等
│   ├── package.json
│   └── vite.config.js
├── run.py              # 仅启动后端的脚本（被 start_all 调用）
└── start_all.py        # 一键启动脚本（安装依赖 + 启动前后端）
```

## 快速上手
1. **克隆仓库**
   ```bash
   git clone https://github.com/your-org/water-quality-platform.git
   cd water-quality-platform
   ```

2. **准备运行环境**
   - Python ≥ 3.9（推荐使用 Conda 或 venv 创建虚拟环境）
   - Node.js ≥ 18（官方 LTS 版本即可，同时自带 npm）

3. **一键启动**
   ```bash
   python platform_demo/start_all.py
   ```
   脚本会自动：
   - 检测并安装 Python 后端依赖
   - 检测 Node/npm，并安装前端依赖
   - 启动 Flask 后端（默认端口 `8000`）
   - 启动 Vite 前端（默认端口 `3000`）
   - 自动打开浏览器访问 `http://localhost:3000`

4. **默认体验账号**
   - 注册/登录入口位于右上角头像菜单
   - SQLite 数据库存放在 `platform_demo/backend/data/users.db`（运行时自动生成，不会提交至仓库）

## 数据使用说明
- 仓库默认加载 `platform_demo/data/sample_water_quality.csv`，这是经过脱敏的样例数据。  
  启动时会直接读取该文件，便于开箱即用。
- 如需替换为自有数据，只需将 CSV 覆盖到同一路径并保持列名一致即可，无需额外配置。

## 安全与隐私
- 仓库已删除所有真实监测数据及训练产物，仅保留用于演示的样例。
- `users.db`、`.env` 等动态文件均被 `.gitignore` 排除，避免泄露凭据。
- 如另有敏感配置（如第三方 API Key），请放入 `.env` 并确保未提交至仓库。

## 本地开发
若需分别运行前后端以便调试：
```bash
# 后端
cd platform_demo
python -m pip install -r backend/requirements.txt
python run.py

# 前端（另开终端）
cd platform_demo/frontend-vue
npm install
npm run dev
```
默认端口可在 `run.py` / `vite.config.js` 中调整。

## 发布到 GitHub 的建议
1. 确认 `git status` 无敏感文件（尤其是 `.csv` 数据、`users.db`、日志等）。
2. 运行 `python platform_demo/start_all.py` 完整验证一次，确保无需手动干预即可成功启动。
3. 如需 CI/CD，可在 GitHub Actions 中执行：
   ```bash
   python -m pip install -r platform_demo/backend/requirements.txt
   npm --prefix platform_demo/frontend-vue install
   npm --prefix platform_demo/frontend-vue run build
   ```
4. 在仓库首页 README 中保留上述快速启动指引，便于其他人复现。

## License
请根据项目实际情况补充许可证声明（MIT、Apache-2.0 等）。

---
如需进一步扩展或集成其他数据源，可在 `backend/app.py` 中调整数据加载逻辑，并配合 `.env` 环境变量管理。欢迎提交 Issue / PR。祝你部署顺利！ 🎉
