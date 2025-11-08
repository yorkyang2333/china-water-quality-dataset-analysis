"""
水质监测可视化平台 - 后端API服务
提供数据查询、预测服务等API接口
"""

from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from functools import wraps
from pathlib import Path

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .models.user import UserManager

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加当前目录到路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.predictor import WaterQualityPredictor
from data.data_service import DataService
from data.quality_analyzer import DataQualityAnalyzer

# 设置静态文件夹（前端文件）
frontend_folder = os.path.join(os.path.dirname(current_dir), 'frontend')
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app, supports_credentials=True)  # 允许跨域请求

print(f"Frontend folder: {frontend_folder}")

if not app.config.get('SECRET_KEY'):
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'water-quality-platform-secret')
app.config.setdefault('TOKEN_EXPIRES_IN', 60 * 60 * 24 * 7)  # 默认7天

serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='user-auth-token')

user_db_path = Path(current_dir) / 'data' / 'users.db'
user_manager = UserManager(user_db_path)
print(f"✓ User database initialized at: {user_db_path}")

# 初始化数据服务和预测器
# 数据文件路径优先级：
# 1. 项目根目录/preprocessed_data_feature_eng.csv
# 2. 项目根目录/preprocess/preprocessed_data_feature_eng.csv
# 3. 项目根目录/preprocessed_data.csv

platform_demo_dir = Path(current_dir).parent
default_data_path = platform_demo_dir / 'data' / 'sample_water_quality.csv'

if not default_data_path.exists():
    raise FileNotFoundError(
        f"数据文件缺失: {default_data_path}\n"
        "请将水质数据CSV放置于 platform_demo/data/ 目录，并命名为 sample_water_quality.csv"
    )

DATA_PATH = str(default_data_path.resolve())
print(f"✓ Data file loaded from: {DATA_PATH}")

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

# 初始化数据质量分析器（使用原始数据集）
ORIGINAL_DATA_PATH = DATA_PATH
try:
    if os.path.exists(ORIGINAL_DATA_PATH):
        quality_analyzer = DataQualityAnalyzer(ORIGINAL_DATA_PATH)
        print(f"✓ Quality Analyzer initialized with original dataset")
    else:
        print(f"⚠ Warning: Original dataset not found at {ORIGINAL_DATA_PATH}")
        quality_analyzer = None
except Exception as e:
    print(f"❌ ERROR initializing Quality Analyzer: {e}")
    import traceback
    traceback.print_exc()
    quality_analyzer = None


def _serialize_user(user: dict) -> dict:
    """统一返回给前端的用户结构"""
    return {
        "id": user.get("id"),
        "username": user.get("username"),
        "email": user.get("email"),
        "full_name": user.get("full_name"),
        "bio": user.get("bio") or "",
        "avatar_url": user.get("avatar_url") or "",
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
    }


def _generate_token(user_id: int) -> str:
    return serializer.dumps({"user_id": user_id})


def _resolve_token() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    # 备用：如果未来需要支持Cookie，可在此扩展
    return None


def _verify_token(token: str) -> int | None:
    try:
        data = serializer.loads(token, max_age=app.config["TOKEN_EXPIRES_IN"])
        return data.get("user_id")
    except SignatureExpired:
        return None
    except BadSignature:
        return None


def login_required(func):
    """校验用户登录状态的装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            return response
        token = _resolve_token()
        if not token:
            return (
                jsonify({"success": False, "error": "未授权访问，请先登录"}),
                401,
            )

        user_id = _verify_token(token)
        if not user_id:
            return (
                jsonify({"success": False, "error": "凭证无效或已过期"}),
                401,
            )

        user = user_manager.get_user_by_id(user_id)
        if user is None:
            return jsonify({"success": False, "error": "用户不存在"}), 401

        g.current_user = user
        return func(*args, **kwargs)

    return wrapper


@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response


@app.after_request
def enrich_response_headers(response):
    origin = request.headers.get('Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Origin', origin)
    response.headers.setdefault(
        'Vary',
        ', '.join(filter(None, [response.headers.get('Vary'), 'Origin']))
    )
    response.headers.setdefault(
        'Access-Control-Allow-Headers',
        'Authorization, Content-Type, X-Requested-With'
    )
    response.headers.setdefault(
        'Access-Control-Allow-Methods',
        'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    )
    response.headers.setdefault('Access-Control-Allow-Credentials', 'true')
    response.headers.setdefault('Content-Type', response.headers.get('Content-Type', 'application/json; charset=utf-8'))
    return response


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


# ------------------------------- 用户管理 ------------------------------- #
@app.route('/api/auth/register', methods=['POST', 'OPTIONS'])
def register_user():
    """用户注册"""
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    payload = request.get_json() or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''
    email = payload.get('email')
    full_name = payload.get('full_name')

    if not username or not password:
        return jsonify({
            'success': False,
            'error': '用户名和密码不能为空'
        }), 400

    try:
        user = user_manager.create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name
        )
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 400
    except Exception as exc:  # 安全兜底
        return jsonify({
            'success': False,
            'error': f'注册失败: {exc}'
        }), 500

    token = _generate_token(user['id'])  # type: ignore[index]
    return jsonify({
        'success': True,
        'data': {
            'token': token,
            'user': _serialize_user(user)
        }
    }), 201


@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def login_user():
    """用户登录"""
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    payload = request.get_json() or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''

    if not username or not password:
        return jsonify({
            'success': False,
            'error': '请输入用户名和密码'
        }), 400

    try:
        user = user_manager.authenticate_user(username, password)
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 401
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': f'登录失败: {exc}'
        }), 500

    token = _generate_token(user['id'])  # type: ignore[index]
    return jsonify({
        'success': True,
        'data': {
            'token': token,
            'user': _serialize_user(user)
        }
    })


@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_profile():
    """获取当前登录用户信息"""
    return jsonify({
        'success': True,
        'data': _serialize_user(g.current_user)
    })


@app.route('/api/user/profile', methods=['PUT'])
@login_required
def update_profile():
    """更新当前用户资料"""
    payload = request.get_json() or {}
    try:
        user = user_manager.update_user(
            g.current_user['id'],  # type: ignore[index]
            email=payload.get('email'),
            full_name=payload.get('full_name'),
            bio=payload.get('bio'),
            avatar_url=payload.get('avatar_url')
        )
    except ValueError as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 400
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': f'更新失败: {exc}'
        }), 500

    return jsonify({
        'success': True,
        'data': _serialize_user(user)
    })

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
    获取数据质量分析结果（基于原始数据集）
    参数：
        - province: 省份（可选）
    """
    try:
        # 优先使用quality_analyzer（基于原始数据集）
        if quality_analyzer is not None:
            province = request.args.get('province')
            print(f"API /monitoring/data-quality called (using original dataset), province={province}")

            # 获取总体指标
            overall_metrics = quality_analyzer.get_overall_metrics()
            print(f"Overall metrics: {overall_metrics}")

            # 获取站点质量数据
            station_quality = quality_analyzer.get_station_quality(province=province)
            print(f"Station quality count: {len(station_quality)}")

            # 获取质量问题
            quality_issues = quality_analyzer.get_quality_issues(province=province)
            print(f"Quality issues count: {len(quality_issues)}")

            # 获取完整度趋势
            trend = quality_analyzer.get_completeness_trend()
            print(f"Trend data points: {len(trend.get('values', []))}")

            return jsonify({
                'success': True,
                'data': {
                    'metrics': overall_metrics,
                    'stations': station_quality,
                    'total': len(station_quality),
                    'issues': quality_issues,
                    'trend': trend
                }
            })

        # 降级使用data_service（如果quality_analyzer不可用）
        else:
            province = request.args.get('province')
            print(f"API /monitoring/data-quality called (using data_service)")
            quality_info = data_service.get_data_quality_info(province)
            return jsonify({
                'success': True,
                'data': quality_info
            })
    except Exception as e:
        print(f"ERROR in /api/monitoring/data-quality: {e}")
        import traceback
        traceback.print_exc()
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

@app.route('/api/statistics/trend', methods=['GET'])
def get_statistics_trend():
    """
    获取统计指标的月度趋势
    "本月"指数据中最后有记录的那个月，"较上月"指与前一个月对比
    用于计算"本月新增"、"较上月变化"等数据
    返回：
      - station_count_change: 站点数量变化 (%)
      - excellent_count_change: 优秀水质变化 (%)
      - warning_stations_change: 预警站点数变化 (%)
      - critical_stations_change: 严重污染站点数变化 (%)
      - current_month: 本月 (格式: YYYY-MM)
      - previous_month: 上月 (格式: YYYY-MM)
    """
    try:
        province = request.args.get('province')

        # 获取上两个月的数据
        df = data_service.df.copy()
        if province:
            df = df[df['Province'] == province]

        if 'Date' not in df.columns:
            return jsonify({
                'success': False,
                'error': 'Date column not found'
            }), 400

        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')

        # 获取最近两个月份（数据中存在的）
        months = sorted(df['YearMonth'].dropna().unique())[-2:]

        if len(months) < 2:
            # 如果没有足够的历史数据，返回默认值
            return jsonify({
                'success': True,
                'data': {
                    'station_count_change': 0,
                    'excellent_count_change': 0,
                    'warning_stations_change': 0,
                    'critical_stations_change': 0,
                    'current_month': 'N/A',
                    'previous_month': 'N/A',
                    'note': '数据不足，无法计算变化趋势'
                }
            })

        current_month = months[-1]
        previous_month = months[-2]

        df_current = df[df['YearMonth'] == current_month]
        df_previous = df[df['YearMonth'] == previous_month]

        # 计算各指标的变化
        # 1. 站点数量变化
        current_stations = df_current['Monitoring_Station'].nunique()
        previous_stations = df_previous['Monitoring_Station'].nunique()
        station_change = ((current_stations - previous_stations) / previous_stations * 100) if previous_stations > 0 else 0

        # 2. 优秀水质数量变化
        current_excellent = len(df_current[df_current['Pollution_Level'] == 'Excellent']) if 'Pollution_Level' in df_current.columns else 0
        previous_excellent = len(df_previous[df_previous['Pollution_Level'] == 'Excellent']) if 'Pollution_Level' in df_previous.columns else 0
        excellent_change = ((current_excellent - previous_excellent) / max(previous_excellent, 1) * 100)

        # 3. 预警站点（Warning级别）变化
        current_warning = len(df_current[df_current['Pollution_Level'] == 'Warning']) if 'Pollution_Level' in df_current.columns else 0
        previous_warning = len(df_previous[df_previous['Pollution_Level'] == 'Warning']) if 'Pollution_Level' in df_previous.columns else 0
        warning_change = ((current_warning - previous_warning) / max(previous_warning, 1) * 100) if previous_warning > 0 else 0

        # 4. 严重污染站点（Critical级别）变化
        current_critical = len(df_current[df_current['Pollution_Level'] == 'Critical']) if 'Pollution_Level' in df_current.columns else 0
        previous_critical = len(df_previous[df_previous['Pollution_Level'] == 'Critical']) if 'Pollution_Level' in df_previous.columns else 0
        critical_change = ((current_critical - previous_critical) / max(previous_critical, 1) * 100) if previous_critical > 0 else 0

        return jsonify({
            'success': True,
            'data': {
                'station_count_change': round(station_change, 1),
                'excellent_count_change': round(excellent_change, 1),
                'warning_stations_change': round(warning_change, 1),
                'critical_stations_change': round(critical_change, 1),
                'current_month': str(current_month),
                'previous_month': str(previous_month)
            }
        })
    except Exception as e:
        print(f"ERROR in /api/statistics/trend: {e}")
        import traceback
        traceback.print_exc()
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
    print("\n  Server running at: http://localhost:8000")
    print("="*60 + "\n")

    # Use environment variable to control debug mode (default: False to avoid double start)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=8000, use_reloader=False)
