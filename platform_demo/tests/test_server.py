"""
测试服务器启动
快速检查服务器是否能正常运行
"""

import sys
import os

# 切换到backend目录
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
os.chdir(backend_dir)

# 添加到路径
sys.path.insert(0, backend_dir)

print("="*60)
print("  Testing Water Quality Platform Server")
print("="*60)
print()

# 测试导入
print("[1/4] Testing imports...")
try:
    from models.predictor import WaterQualityPredictor
    from data.data_service import DataService
    print("  OK - All modules imported successfully")
except Exception as e:
    print(f"  ERROR - Import failed: {e}")
    sys.exit(1)

# 测试数据加载
print()
print("[2/4] Testing data loading...")
try:
    # 获取项目根目录的预处理数据
    platform_demo_dir = os.path.dirname(backend_dir)
    project_root = os.path.dirname(platform_demo_dir)
    data_path = os.path.join(project_root, 'preprocessed_data.csv')

    print(f"  Using data file: {data_path}")
    if not os.path.exists(data_path):
        print("  ERROR - Data file not found!")
        print(f"  Please ensure preprocessed_data.csv exists in: {project_root}")
        sys.exit(1)

    data_service = DataService(data_path)
    print(f"  OK - Data loaded: {len(data_service.df)} records")
except Exception as e:
    print(f"  ERROR - Data loading failed: {e}")
    sys.exit(1)

# 测试预测器
print()
print("[3/4] Testing predictor...")
try:
    predictor = WaterQualityPredictor()
    print(f"  OK - Predictor initialized")
except Exception as e:
    print(f"  ERROR - Predictor failed: {e}")
    sys.exit(1)

# 启动Flask服务器
print()
print("[4/4] Starting Flask server...")
print()
print("="*60)
print("  Server will start at: http://localhost:5000")
print("  Press Ctrl+C to stop")
print("="*60)
print()

try:
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)
except Exception as e:
    print(f"ERROR - Server failed to start: {e}")
    sys.exit(1)
