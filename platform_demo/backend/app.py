"""
水质监测可视化平台 - 后端API服务
提供数据查询、预测服务等API接口
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# 添加当前目录到路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.predictor import WaterQualityPredictor
from data.data_service import DataService

# 设置静态文件夹（前端文件）
frontend_folder = os.path.join(os.path.dirname(current_dir), 'frontend')
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)  # 允许跨域请求

print(f"Frontend folder: {frontend_folder}")

# 初始化数据服务和预测器
# 数据文件在项目根目录的上一级
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'preprocessed_data_feature_eng.csv'
)

# 检查数据文件是否存在
print(f"Looking for data file at: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: Data file not found at {DATA_PATH}")
    print(f"Please ensure preprocessed_data_feature_eng.csv is in the correct location")
else:
    print(f"✓ Data file found")

try:
    data_service = DataService(DATA_PATH)
    print(f"✓ DataService initialized successfully")
except Exception as e:
    print(f"❌ ERROR initializing DataService: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    predictor = WaterQualityPredictor()
    print(f"✓ Predictor initialized successfully")
except Exception as e:
    print(f"❌ ERROR initializing Predictor: {e}")
    import traceback
    traceback.print_exc()
    raise

@app.route('/')
def index():
    """返回前端主页"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    """返回CSS文件"""
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """返回JS文件"""
    return send_from_directory(os.path.join(app.static_folder, 'js'), filename)

@app.route('/api/stations', methods=['GET'])
def get_stations():
    """获取所有监测站点信息"""
    try:
        print("API /stations called")
        stations = data_service.get_all_stations()
        print(f"Returning {len(stations)} stations")
        return jsonify({
            'success': True,
            'data': stations
        })
    except Exception as e:
        print(f"ERROR in /api/stations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/timeline', methods=['GET'])
def get_timeline_data():
    """
    获取时间轴数据
    参数：
        - start_date: 开始日期 (YYYY-MM-DD)
        - end_date: 结束日期 (YYYY-MM-DD)
        - station: 站点名称（可选）
        - province: 省份名称（可选）
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        station = request.args.get('station')
        province = request.args.get('province')

        data = data_service.get_timeline_data(
            start_date=start_date,
            end_date=end_date,
            station=station,
            province=province
        )

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/spatial', methods=['POST'])
def get_spatial_data():
    """
    空间查询数据
    参数：
        - type: 'rectangle' 或 'circle'
        - bounds: 边界坐标 (rectangle: [[lat1,lng1], [lat2,lng2]])
        - center: 圆心坐标 (circle: [lat, lng])
        - radius: 半径（米）(circle only)
        - start_date: 开始日期（可选）
        - end_date: 结束日期（可选）
    """
    try:
        query = request.get_json()
        data = data_service.get_spatial_data(query)

        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/pollution_level', methods=['POST'])
def predict_pollution_level():
    """
    预测污染等级
    输入水质指标，预测污染等级
    """
    try:
        input_data = request.get_json()
        result = predictor.predict_pollution_level(input_data)

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/water_quality', methods=['POST'])
def predict_water_quality():
    """
    预测水质指标
    输入地点+时间，预测未来水质
    """
    try:
        input_data = request.get_json()
        result = predictor.predict_water_quality(input_data)

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/statistics/summary', methods=['GET'])
def get_statistics_summary():
    """获取统计摘要信息"""
    try:
        province = request.args.get('province')
        print(f"API /statistics/summary called, province={province}")
        summary = data_service.get_statistics_summary(province)
        print(f"Statistics summary generated successfully")

        return jsonify({
            'success': True,
            'data': summary
        })
    except Exception as e:
        print(f"ERROR in /api/statistics/summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """
    获取趋势数据
    参数：
        - indicator: 指标名称
        - province: 省份（可选）
        - station: 站点（可选）
        - interval: 聚合间隔 (day/week/month)
    """
    try:
        indicator = request.args.get('indicator', 'Water_Quality_Index')
        province = request.args.get('province')
        station = request.args.get('station')
        interval = request.args.get('interval', 'month')

        trends = data_service.get_trends(
            indicator=indicator,
            province=province,
            station=station,
            interval=interval
        )

        return jsonify({
            'success': True,
            'data': trends
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/date-range', methods=['GET'])
def get_date_range():
    """
    获取数据的日期范围
    返回最小和最大日期
    """
    try:
        date_range = data_service.get_date_range()
        return jsonify({
            'success': True,
            'data': date_range
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/by-date', methods=['GET'])
def get_data_by_date():
    """
    获取截止到指定日期的站点数据
    参数：
        - date: 截止日期 (YYYY-MM-DD)
    """
    try:
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({
                'success': False,
                'error': 'Date parameter is required'
            }), 400

        result = data_service.get_stations_by_date(date_str)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============= GIS检测功能 API =============

@app.route('/api/monitoring/realtime', methods=['GET'])
def get_realtime_monitoring():
    """
    获取实时监测指标
    返回关键水质指标的实时统计
    """
    try:
        province = request.args.get('province')
        data = data_service.get_realtime_monitoring(province)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitoring/alerts', methods=['GET'])
def get_monitoring_alerts():
    """
    获取告警信息
    返回超标站点和异常数据
    """
    try:
        province = request.args.get('province')
        alerts = data_service.get_monitoring_alerts(province)
        return jsonify({
            'success': True,
            'data': alerts
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitoring/data-quality', methods=['GET'])
def get_data_quality():
    """
    获取数据质量信息
    返回数据完整性、更新频率等统计
    """
    try:
        province = request.args.get('province')
        quality_info = data_service.get_data_quality_info(province)
        return jsonify({
            'success': True,
            'data': quality_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/pollution-distribution', methods=['GET'])
def get_pollution_distribution():
    """
    获取污染等级分布数据（用于饼图）
    """
    try:
        province = request.args.get('province')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        distribution = data_service.get_pollution_distribution(
            province=province,
            start_date=start_date,
            end_date=end_date
        )
        return jsonify({
            'success': True,
            'data': distribution
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/province-comparison', methods=['GET'])
def get_province_comparison():
    """
    获取省份对比数据（用于柱状图）
    """
    try:
        indicator = request.args.get('indicator', 'Water_Quality_Index')
        comparison_data = data_service.get_province_comparison(indicator)
        return jsonify({
            'success': True,
            'data': comparison_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/indicator-correlation', methods=['GET'])
def get_indicator_correlation():
    """
    获取指标相关性数据
    """
    try:
        province = request.args.get('province')
        correlation_data = data_service.get_indicator_correlation(province)
        return jsonify({
            'success': True,
            'data': correlation_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/multi-indicator-trend', methods=['GET'])
def get_multi_indicator_trend():
    """
    获取多指标趋势数据
    """
    try:
        indicators = request.args.getlist('indicators')
        if not indicators:
            indicators = ['Water_Quality_Index', 'COD_mg_L', 'pH', 'Total_Nitrogen_mg_L']

        province = request.args.get('province')
        interval = request.args.get('interval', 'month')

        trend_data = data_service.get_multi_indicator_trend(
            indicators=indicators,
            province=province,
            interval=interval
        )
        return jsonify({
            'success': True,
            'data': trend_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Water Quality Monitoring Platform - Backend Server")
    print("="*60)
    print(f"\n  Data loaded: {len(data_service.df)} records")
    print(f"  Provinces: {data_service.df['Province'].nunique()}")
    print(f"  Stations: {data_service.df['Monitoring_Station'].nunique()}")
    print("\n  Server running at: http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
