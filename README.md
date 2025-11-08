# 水质监测平台（精简开源版）

该仓库提供一个可在本地或服务器快速部署的 **水质监测数据可视化平台**。  
仓库中已移除所有可能泄露隐私的原始数据与账号信息，并内置一份仅用于演示的样例数据，便于在任何环境下一键运行。

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
