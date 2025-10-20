import pandas as pd
import os
import folium
from folium import FeatureGroup, Element
import json
import requests

# 定义新的输出目录
OUTPUT_DIR = 'Outputs/EDA_Map'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GeoJSON文件URL
GEOJSON_URL = 'https://raw.githubusercontent.com/longwosion/geojson-map-china/master/china.json'

# 英汉省份名称转换字典
PROVINCE_NAME_MAP = {
    'Beijing': '北京',
    'Guangdong': '广东',
    'Henan': '河南',
    'Hubei': '湖北',
    'Jiangsu': '江苏',
    'Shandong': '山东',
    'Shanghai': '上海',
    'Sichuan': '四川',
    'Yunnan': '云南',
    'Zhejiang': '浙江'
}

def normalize_province_name(name):
    """标准化省份名称，去除常见后缀以提高匹配成功率"""
    if not isinstance(name, str):
        return ""
    suffixes = ['省', '市', '自治区', '壮族自治区', '回族自治区', '维吾尔自治区', '特别行政区']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    return name

def load_data(filepath):
    """加载经过特征工程处理的数据"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df

def get_geojson_data(url):
    """从URL获取GeoJSON数据并提取省份中心点和标准化名称"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        geo_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GeoJSON from {url}: {e}")
        return None, None

    province_centers = {}
    # 在GeoJSON中直接添加标准化的名称，以便folium使用
    for feature in geo_data.get('features', []):
        properties = feature.get('properties', {})
        province_name = properties.get('name')
        if province_name:
            normalized_name = normalize_province_name(province_name)
            properties['normalized_name'] = normalized_name # 新增字段
            if 'cp' in properties:
                lon, lat = properties['cp']
                province_centers[normalized_name] = [lat, lon]
    return geo_data, province_centers

def create_province_choropleth_map(df, geo_data, province_centers):
    """创建现代化的交互式水质地图，使用动态重着色而非图层切换"""
    print("CREATING PROVINCE-LEVEL WATER QUALITY MAP WITH CUSTOM UI")

    # 1. 定义要可视化的指标及其配置
    indicators_to_visualize = {
        'Water_Quality_Index': {
            'name': '水质指数',
            'name_en': 'WQI',
            'color': 'YlOrRd',
            'description': '综合评估水体质量的指数',
            'unit': '指数值',
            'icon': '💧'
        },
        'Total_Heavy_Metals_ug_L': {
            'name': '总重金属',
            'name_en': 'Heavy Metals',
            'color': 'Reds',
            'description': '铅、镉、汞的总和',
            'unit': 'μg/L',
            'icon': '⚠️'
        },
        'COD_mg_L': {
            'name': '化学需氧量',
            'name_en': 'COD',
            'color': 'Oranges',
            'description': '水中有机物污染程度',
            'unit': 'mg/L',
            'icon': '🧪'
        },
        'Total_Nitrogen_mg_L': {
            'name': '总氮',
            'name_en': 'Total N',
            'color': 'Purples',
            'description': '水体富营养化关键指标',
            'unit': 'mg/L',
            'icon': '🌊'
        },
        'Total_Phosphorus_mg_L': {
            'name': '总磷',
            'name_en': 'Total P',
            'color': 'BuPu',
            'description': '水体富营养化关键指标',
            'unit': 'mg/L',
            'icon': '🌿'
        },
        'Dissolved_Oxygen_mg_L': {
            'name': '溶解氧',
            'name_en': 'DO',
            'color': 'GnBu_r',
            'description': '水中氧气含量，越高越好',
            'unit': 'mg/L',
            'icon': '💨'
        },
        'pH': {
            'name': 'pH值',
            'name_en': 'pH',
            'color': 'RdYlBu',
            'description': '酸碱度，pH=7为中性，越低越酸，越高越碱',
            'unit': 'pH',
            'icon': '⚗️'
        },
        'Ammonia_N_mg_L': {
            'name': '氨氮',
            'name_en': 'Ammonia-N',
            'color': 'YlGn',
            'description': '水体污染的重要指标，值越低越好',
            'unit': 'mg/L',
            'icon': '🔬'
        },
        'Turbidity_NTU': {
            'name': '浊度',
            'name_en': 'Turbidity',
            'color': 'Greys',
            'description': '水的清澈程度，值越低水越清澈',
            'unit': 'NTU',
            'icon': '🌫️'
        },
        'COD_BOD_Ratio': {
            'name': 'COD/BOD比值',
            'name_en': 'COD/BOD Ratio',
            'color': 'PuRd',
            'description': '反映有机物可生物降解性，比值越高越难降解',
            'unit': '比值',
            'icon': '📊'
        }
    }

    # 2. 按省份聚合所有需要的指标
    agg_dict = {indicator: 'mean' for indicator in indicators_to_visualize}
    agg_dict['Pollution_Level'] = lambda x: x.value_counts().index[0]
    agg_dict['Monitoring_Station'] = 'nunique'

    province_agg = df.groupby('Province').agg(agg_dict).reset_index()
    province_agg.rename(columns={
        'Pollution_Level': 'Most_Common_Pollution_Level',
        'Monitoring_Station': 'Station_Count'
    }, inplace=True)

    # 3. 添加中文名称和标准化中文名称列
    province_agg['Province_CN'] = province_agg['Province'].map(PROVINCE_NAME_MAP)
    province_agg['Province_CN_Normalized'] = province_agg['Province_CN'].apply(normalize_province_name)

    # 4. 计算全国统计数据
    total_stations = df['Monitoring_Station'].nunique()
    total_records = len(df)
    avg_wqi = df['Water_Quality_Index'].mean()

    # 计算每个指标的全国平均值
    national_averages = {
        indicator: df[indicator].mean()
        for indicator in indicators_to_visualize.keys()
    }

    # 计算污染等级分布
    pollution_distribution = df['Pollution_Level'].value_counts().to_dict()

    # 5. 创建地图基础
    map_center = [35.8617, 104.1954]
    m = folium.Map(location=map_center, zoom_start=5, tiles=None)
    folium.TileLayer('CartoDB positron', control=False).add_to(m)

    # 6. 准备所有指标的数据
    print("\nPreparing data for all indicators...")

    # 定义不同颜色方案的实际颜色值
    color_schemes = {
        'YlOrRd': ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'],
        'Reds': ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'],
        'Oranges': ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#8c2d04'],
        'Purples': ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'],
        'BuPu': ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b'],
        'GnBu_r': ['#084081', '#0868ac', '#2b8cbe', '#4eb3d3', '#7bccc4', '#a8ddb5', '#ccebc5', '#f0f9e8'],
        'RdYlBu': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'],  # pH (红-黄-蓝)
        'YlGn': ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32'],  # 氨氮 (黄-绿)
        'Greys': ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525'],  # 浊度 (灰色)
        'PuRd': ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f'],  # COD/BOD比值 (紫-红)
    }

    # 准备每个省份的所有指标数据
    province_data_all = {}
    layer_metadata = []

    for i, (indicator, config) in enumerate(indicators_to_visualize.items()):
        min_val = province_agg[indicator].min()
        max_val = province_agg[indicator].max()

        colors = color_schemes.get(config['color'], color_schemes['YlOrRd'])

        # 存储图层元数据
        layer_metadata.append({
            'min': float(min_val),
            'max': float(max_val),
            'avg': float(national_averages[indicator]),
            'unit': config['unit'],
            'colors': colors
        })

        # 为每个省份准备数据
        for _, row in province_agg.iterrows():
            normalized_name = row['Province_CN_Normalized']
            if pd.isna(normalized_name):
                continue

            if normalized_name not in province_data_all:
                province_data_all[normalized_name] = {}

            province_data_all[normalized_name][indicator] = float(row[indicator])

        print(f" - Prepared data for '{config['name']}' | Range: {min_val:.2f} - {max_val:.2f}")

    # 7. 创建单个GeoJSON图层（不使用多图层）
    print("\nCreating map with single interactive layer...")

    # 默认使用第一个指标
    first_indicator = list(indicators_to_visualize.keys())[0]
    first_config = indicators_to_visualize[first_indicator]
    first_colors = color_schemes[first_config['color']]
    first_min = province_agg[first_indicator].min()
    first_max = province_agg[first_indicator].max()

    import branca.colormap as cm
    first_colormap = cm.LinearColormap(
        colors=first_colors,
        vmin=first_min,
        vmax=first_max
    )

    def style_function(feature):
        province_name = normalize_province_name(feature['properties'].get('name', ''))
        value = province_data_all.get(province_name, {}).get(first_indicator, 0)
        return {
            'fillColor': first_colormap(value) if value else '#e0e0e0',
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
            'className': 'water-quality-polygon'
        }

    # 添加GeoJson层
    geojson_layer = folium.GeoJson(
        geo_data,
        style_function=style_function,
        highlight_function=lambda x: {'weight': 2, 'fillOpacity': 0.9}
    )
    geojson_layer.add_to(m)

    print(" - Single interactive layer created successfully")

    # 7. 创建省份信息标记 - 始终显示
    for _, row in province_agg.iterrows():
        if pd.isna(row['Province_CN_Normalized']):
            continue
        center_coords = province_centers.get(row['Province_CN_Normalized'])
        if center_coords:
            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 280px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 15px; margin: -10px -10px 10px -10px;
                            border-radius: 8px 8px 0 0;">
                    <h3 style="margin: 0; font-size: 20px;">{row['Province_CN']}</h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 12px;">
                        监测站数量: {row['Station_Count']} 个
                    </p>
                </div>
                <div style="padding: 5px 0;">
                    <div style="background: #f8f9fa; padding: 8px; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #495057;">主要污染等级:</strong>
                        <span style="color: #e83e8c; font-weight: bold;">{row['Most_Common_Pollution_Level']}</span>
                    </div>
            """

            for indicator, config in indicators_to_visualize.items():
                value = row[indicator]
                popup_html += f"""
                    <div style="display: flex; justify-content: space-between;
                                padding: 6px 0; border-bottom: 1px solid #e9ecef;">
                        <span style="color: #6c757d;">
                            {config['icon']} {config['name']}:
                        </span>
                        <span style="font-weight: bold; color: #212529;">
                            {value:.2f} {config['unit']}
                        </span>
                    </div>
                """

            popup_html += "</div></div>"

            folium.Marker(
                location=center_coords,
                popup=folium.Popup(popup_html, max_width=350),
                icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
            ).add_to(m)

    # 8. 准备排名数据
    rankings_data = {}
    for indicator, config in indicators_to_visualize.items():
        sorted_provinces = province_agg.sort_values(by=indicator)
        # 对于溶解氧,值越高越好;对于其他指标,值越低越好
        if indicator == 'Dissolved_Oxygen_mg_L':
            top3 = sorted_provinces.tail(3)[['Province_CN', indicator]].values.tolist()[::-1]
            bottom3 = sorted_provinces.head(3)[['Province_CN', indicator]].values.tolist()
        else:
            top3 = sorted_provinces.head(3)[['Province_CN', indicator]].values.tolist()
            bottom3 = sorted_provinces.tail(3)[['Province_CN', indicator]].values.tolist()[::-1]
        rankings_data[indicator] = {'best': top3, 'worst': bottom3}

    # 9. 构建自定义HTML界面 - 包含丰富的可视化元素
    custom_html = f"""
    <!DOCTYPE html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

            body {{
                font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 0;
            }}

            /* 标题栏 */
            .header-panel {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                z-index: 1000;
            }}

            .header-panel h1 {{
                margin: 0;
                font-size: 24px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }}

            .header-panel p {{
                margin: 3px 0 0 0;
                opacity: 0.95;
                font-size: 12px;
                font-weight: 300;
            }}

            /* 图层切换器 */
            .layer-switcher {{
                position: fixed;
                top: 90px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                border-radius: 15px;
                padding: 10px 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.15);
                z-index: 999;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                max-width: 90vw;
                overflow-x: auto;
                justify-content: center;
            }}

            .layer-btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 25px;
                background: #f5f7fa;
                color: #4a5568;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                white-space: nowrap;
                font-family: 'Poppins', sans-serif;
            }}

            .layer-btn:hover {{
                background: #e9ecef;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}

            .layer-btn.active {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 16px rgba(102,126,234,0.4);
            }}

            /* 左侧面板容器 */
            .left-panels-container {{
                position: fixed;
                top: 180px;
                left: 20px;
                width: 280px;
                max-height: calc(100vh - 200px);
                overflow-y: auto;
                z-index: 999;
            }}

            /* 面板通用样式 */
            .panel {{
                background: white;
                border-radius: 15px;
                padding: 18px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }}

            .panel-title {{
                font-size: 15px;
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 2px solid #667eea;
            }}

            /* 数据统计面板 */
            .stat-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 9px 11px;
                margin: 6px 0;
                background: linear-gradient(135deg, #f6f8fb 0%, #f1f3f6 100%);
                border-radius: 10px;
                transition: all 0.3s ease;
            }}

            .stat-item:hover {{
                transform: translateX(5px);
                box-shadow: 0 4px 12px rgba(102,126,234,0.15);
            }}

            .stat-label {{
                font-size: 12px;
                color: #4a5568;
                font-weight: 500;
            }}

            .stat-value {{
                font-size: 15px;
                font-weight: 700;
                color: #667eea;
            }}

            .stat-icon {{
                font-size: 17px;
                margin-right: 6px;
            }}

            /* 颜色图例 */
            .color-legend {{
                margin-top: 10px;
            }}

            .legend-gradient {{
                height: 20px;
                border-radius: 10px;
                margin: 10px 0;
                transition: background 0.3s ease;
            }}

            .legend-labels {{
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #4a5568;
                margin-top: 5px;
            }}

            .legend-avg {{
                margin-top: 10px;
                padding: 8px;
                background: #fff3cd;
                border-left: 3px solid #ffc107;
                border-radius: 5px;
                font-size: 12px;
            }}

            /* 省份排名 */
            .ranking-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 10px;
                margin: 5px 0;
                background: #f8f9fa;
                border-radius: 8px;
                font-size: 12px;
            }}

            .ranking-item.best {{
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-left: 3px solid #28a745;
            }}

            .ranking-item.worst {{
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border-left: 3px solid #dc3545;
            }}

            .rank-medal {{
                font-size: 16px;
                margin-right: 6px;
            }}

            .rank-province {{
                flex: 1;
                font-weight: 500;
                color: #2d3748;
            }}

            .rank-value {{
                font-weight: 700;
                color: #4a5568;
            }}

            /* 污染分布图 */
            .pollution-chart {{
                margin-top: 12px;
            }}

            .chart-bar {{
                margin: 8px 0;
            }}

            .chart-bar-label {{
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #4a5568;
                margin-bottom: 4px;
            }}

            .chart-bar-fill {{
                height: 20px;
                border-radius: 10px;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}

            .chart-bar-fill::after {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 2s infinite;
            }}

            @keyframes shimmer {{
                100% {{
                    left: 100%;
                }}
            }}

            /* 右下角面板 */
            .current-indicator-panel {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                z-index: 999;
                width: 320px;
            }}

            .indicator-title {{
                font-size: 18px;
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
            }}

            .indicator-icon {{
                font-size: 24px;
                margin-right: 10px;
            }}

            .indicator-desc {{
                font-size: 13px;
                color: #4a5568;
                line-height: 1.6;
                margin-bottom: 12px;
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
            }}

            .usage-hint {{
                font-size: 11px;
                color: #718096;
                padding-top: 10px;
                border-top: 1px solid #e2e8f0;
            }}

            /* 地图容器 */
            #map {{
                position: fixed;
                top: 75px;
                left: 0;
                right: 0;
                bottom: 0;
            }}

            /* 隐藏默认控制器 */
            .leaflet-control-layers {{
                display: none !important;
            }}

            /* 滚动条美化 */
            .left-panels-container::-webkit-scrollbar {{
                width: 6px;
            }}

            .left-panels-container::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 10px;
            }}

            .left-panels-container::-webkit-scrollbar-thumb {{
                background: #667eea;
                border-radius: 10px;
            }}

            /* 响应式设计 */
            @media (max-width: 1200px) {{
                .left-panels-container {{
                    width: 240px;
                }}

                .current-indicator-panel {{
                    width: 280px;
                }}
            }}

            @media (max-width: 768px) {{
                .left-panels-container {{
                    top: 150px;
                    left: 10px;
                    width: calc(50% - 15px);
                    max-height: calc(100vh - 170px);
                }}

                .current-indicator-panel {{
                    bottom: 10px;
                    right: 10px;
                    width: calc(50% - 15px);
                }}

                .layer-switcher {{
                    top: 75px;
                    padding: 6px 10px;
                    gap: 5px;
                }}

                .layer-btn {{
                    padding: 8px 14px;
                    font-size: 11px;
                }}

                .header-panel h1 {{
                    font-size: 18px;
                }}
            }}
        </style>
    </head>
    <body>
        <!-- 标题栏 -->
        <div class="header-panel">
            <h1>🇨🇳 中国水质监测分析平台</h1>
            <p>China Water Quality Monitoring & Analysis Dashboard</p>
        </div>

        <!-- 图层切换器 -->
        <div class="layer-switcher" id="layerSwitcher">
    """

    # 生成图层按钮
    for i, (indicator, config) in enumerate(indicators_to_visualize.items()):
        active_class = "active" if i == 0 else ""
        custom_html += f"""
            <button class="layer-btn {active_class}" onclick="switchLayer({i})" data-layer="{i}">
                {config['icon']} {config['name']}
            </button>
        """

    # 计算污染等级百分比
    total_pollution_records = sum(pollution_distribution.values())
    pollution_percentages = {
        level: (count / total_pollution_records * 100)
        for level, count in pollution_distribution.items()
    }

    # 污染等级颜色映射
    pollution_colors = {
        'Excellent': '#28a745',
        'Good': '#5cb85c',
        'Fair': '#ffc107',
        'Poor': '#fd7e14',
        'Very Poor': '#dc3545'
    }

    custom_html += f"""
        </div>

        <!-- 左侧面板容器 -->
        <div class="left-panels-container">
            <!-- 数据概览面板 -->
            <div class="panel">
                <div class="panel-title">📊 数据概览</div>
                <div class="stat-item">
                    <div>
                        <span class="stat-icon">🗺️</span>
                        <span class="stat-label">覆盖省份</span>
                    </div>
                    <span class="stat-value">{len(province_agg)}</span>
                </div>
                <div class="stat-item">
                    <div>
                        <span class="stat-icon">📍</span>
                        <span class="stat-label">监测站点</span>
                    </div>
                    <span class="stat-value">{total_stations}</span>
                </div>
                <div class="stat-item">
                    <div>
                        <span class="stat-icon">📝</span>
                        <span class="stat-label">数据记录</span>
                    </div>
                    <span class="stat-value">{total_records:,}</span>
                </div>
                <div class="stat-item">
                    <div>
                        <span class="stat-icon">💧</span>
                        <span class="stat-label">平均水质指数</span>
                    </div>
                    <span class="stat-value">{avg_wqi:.1f}</span>
                </div>
            </div>

            <!-- 颜色图例面板 -->
            <div class="panel">
                <div class="panel-title">🎨 颜色图例</div>
                <div class="color-legend">
                    <div class="legend-gradient" id="legendGradient"></div>
                    <div class="legend-labels">
                        <span id="legendMin">{layer_metadata[0]['min']:.1f}</span>
                        <span id="legendMax">{layer_metadata[0]['max']:.1f}</span>
                    </div>
                    <div class="legend-avg">
                        <strong>全国平均:</strong>
                        <span id="legendAvg">{layer_metadata[0]['avg']:.2f}</span>
                        <span id="legendUnit">{list(indicators_to_visualize.values())[0]['unit']}</span>
                    </div>
                </div>
            </div>

            <!-- 省份排名面板 -->
            <div class="panel">
                <div class="panel-title">🏆 省份排名</div>
                <div id="rankingContent">
                    <div style="font-size: 13px; font-weight: 600; color: #28a745; margin: 10px 0 5px 0;">✅ 表现最佳</div>
    """

    # 添加初始排名（第一个指标）
    first_indicator = list(indicators_to_visualize.keys())[0]
    first_ranking = rankings_data[first_indicator]

    medals = ['🥇', '🥈', '🥉']
    for i, (province, value) in enumerate(first_ranking['best']):
        custom_html += f"""
                    <div class="ranking-item best">
                        <span class="rank-medal">{medals[i]}</span>
                        <span class="rank-province">{province}</span>
                        <span class="rank-value">{value:.1f}</span>
                    </div>
        """

    custom_html += """
                    <div style="font-size: 13px; font-weight: 600; color: #dc3545; margin: 15px 0 5px 0;">⚠️ 需要改善</div>
    """

    for i, (province, value) in enumerate(first_ranking['worst']):
        custom_html += f"""
                    <div class="ranking-item worst">
                        <span class="rank-medal">{medals[i]}</span>
                        <span class="rank-province">{province}</span>
                        <span class="rank-value">{value:.1f}</span>
                    </div>
        """

    custom_html += f"""
                </div>
            </div>

            <!-- 污染等级分布面板 -->
            <div class="panel">
                <div class="panel-title">📈 污染等级分布</div>
                <div class="pollution-chart">
    """

    # 添加污染等级条形图
    for level, percentage in sorted(pollution_percentages.items(), key=lambda x: x[1], reverse=True):
        color = pollution_colors.get(level, '#6c757d')
        count = pollution_distribution[level]
        custom_html += f"""
                    <div class="chart-bar">
                        <div class="chart-bar-label">
                            <span>{level}</span>
                            <span>{percentage:.1f}% ({count:,}条)</span>
                        </div>
                        <div class="chart-bar-fill" style="width: {percentage}%; background: {color};"></div>
                    </div>
        """

    custom_html += f"""
                </div>
            </div>
        </div>

        <!-- 当前指标说明面板 -->
        <div class="current-indicator-panel">
            <div class="indicator-title">
                <span class="indicator-icon" id="currentIcon">{list(indicators_to_visualize.values())[0]['icon']}</span>
                <span id="currentName">{list(indicators_to_visualize.values())[0]['name']}</span>
            </div>
            <div class="indicator-desc" id="currentDesc">
                {list(indicators_to_visualize.values())[0]['description']}
                <br><strong>单位:</strong> {list(indicators_to_visualize.values())[0]['unit']}
            </div>
            <div class="usage-hint">
                💡 <strong>操作提示:</strong> 点击上方按钮切换不同水质指标图层。点击地图标记查看省份详细数据。颜色越深表示数值越高。
            </div>
        </div>

        <script>
            // 省份数据 - 所有指标
            const provinceDataAll = {json.dumps(province_data_all)};

            // 指标名称列表
            const indicatorKeys = {json.dumps(list(indicators_to_visualize.keys()))};

            // 指标配置数据
            const indicators = {json.dumps([{
                'name': config['name'],
                'icon': config['icon'],
                'description': config['description'],
                'unit': config['unit']
            } for config in indicators_to_visualize.values()])};

            // 图层元数据
            const layerMetadata = {json.dumps(layer_metadata)};

            // 排名数据
            const rankingsData = {json.dumps(rankings_data)};

            // 颜色插值函数
            function interpolateColor(colors, value, min, max) {{
                if (value <= min) return colors[0];
                if (value >= max) return colors[colors.length - 1];

                const ratio = (value - min) / (max - min);
                const index = ratio * (colors.length - 1);
                const lowerIndex = Math.floor(index);
                const upperIndex = Math.ceil(index);

                if (lowerIndex === upperIndex) return colors[lowerIndex];

                const lowerColor = colors[lowerIndex];
                const upperColor = colors[upperIndex];
                const localRatio = index - lowerIndex;

                // 解析颜色
                function parseColor(hex) {{
                    const r = parseInt(hex.slice(1, 3), 16);
                    const g = parseInt(hex.slice(3, 5), 16);
                    const b = parseInt(hex.slice(5, 7), 16);
                    return [r, g, b];
                }}

                const [r1, g1, b1] = parseColor(lowerColor);
                const [r2, g2, b2] = parseColor(upperColor);

                const r = Math.round(r1 + (r2 - r1) * localRatio);
                const g = Math.round(g1 + (g2 - g1) * localRatio);
                const b = Math.round(b1 + (b2 - b1) * localRatio);

                return `#${{r.toString(16).padStart(2, '0')}}${{g.toString(16).padStart(2, '0')}}${{b.toString(16).padStart(2, '0')}}`;
            }}

            // 图层切换函数 - 使用DOM重着色
            function switchLayer(layerIndex) {{
                console.log('🔄 切换到指标:', layerIndex);

                // 更新按钮状态
                const buttons = document.querySelectorAll('.layer-btn');
                buttons.forEach((btn, idx) => {{
                    if (idx === layerIndex) {{
                        btn.classList.add('active');
                    }} else {{
                        btn.classList.remove('active');
                    }}
                }});

                // 更新当前指标信息
                const indicator = indicators[layerIndex];
                document.getElementById('currentIcon').textContent = indicator.icon;
                document.getElementById('currentName').textContent = indicator.name;
                document.getElementById('currentDesc').innerHTML =
                    indicator.description + '<br><strong>单位:</strong> ' + indicator.unit;

                // 更新颜色图例
                const metadata = layerMetadata[layerIndex];
                document.getElementById('legendMin').textContent = metadata.min.toFixed(1);
                document.getElementById('legendMax').textContent = metadata.max.toFixed(1);
                document.getElementById('legendAvg').textContent = metadata.avg.toFixed(2);
                document.getElementById('legendUnit').textContent = metadata.unit;

                // 更新颜色渐变条
                if (metadata.colors && metadata.colors.length > 0) {{
                    const gradient = 'linear-gradient(to right, ' + metadata.colors.join(', ') + ')';
                    const gradientElement = document.getElementById('legendGradient');
                    if (gradientElement) {{
                        gradientElement.style.background = gradient;
                        console.log('✅ 图例颜色已更新');
                    }}
                }}

                // 更新排名
                updateRankings(layerIndex);

                // 重新着色地图上的多边形
                const indicatorKey = indicatorKeys[layerIndex];
                const colors = metadata.colors;
                const minVal = metadata.min;
                const maxVal = metadata.max;

                console.log('🎨 开始重新着色地图...');
                let polygonsRecolored = 0;

                // 找到所有的path元素
                const allPaths = document.querySelectorAll('path');
                console.log('  找到', allPaths.length, '个path元素');

                allPaths.forEach(path => {{
                    // 尝试找到该path对应的省份名称
                    // 方法: 从父元素的属性或数据中获取
                    let provinceName = null;

                    // 检查是否有data-province属性
                    if (path.dataset && path.dataset.province) {{
                        provinceName = path.dataset.province;
                    }}

                    // 如果没有找到省份名称，暂时跳过
                    if (!provinceName) {{
                        return;
                    }}

                    // 获取该省份该指标的值
                    const provinceData = provinceDataAll[provinceName];
                    if (!provinceData || !provinceData[indicatorKey]) {{
                        return;
                    }}

                    const value = provinceData[indicatorKey];
                    const color = interpolateColor(colors, value, minVal, maxVal);

                    // 应用新颜色
                    path.style.fill = color;
                    polygonsRecolored++;
                }});

                console.log('✅ 重新着色完成:', polygonsRecolored, '个多边形');

                if (polygonsRecolored === 0) {{
                    console.warn('⚠️ 没有任何多边形被重新着色');
                    console.log('💡 可能需要为GeoJSON元素添加省份标识');
                }}
            }}

            // 更新排名显示
            function updateRankings(layerIndex) {{
                const indicatorKeys = {json.dumps(list(indicators_to_visualize.keys()))};
                const rankings = rankingsData[indicatorKeys[layerIndex]];
                const medals = ['🥇', '🥈', '🥉'];

                let html = '<div style="font-size: 13px; font-weight: 600; color: #28a745; margin: 10px 0 5px 0;">✅ 表现最佳</div>';

                rankings.best.forEach((item, i) => {{
                    html += `
                        <div class="ranking-item best">
                            <span class="rank-medal">${{medals[i]}}</span>
                            <span class="rank-province">${{item[0]}}</span>
                            <span class="rank-value">${{item[1].toFixed(1)}}</span>
                        </div>
                    `;
                }});

                html += '<div style="font-size: 13px; font-weight: 600; color: #dc3545; margin: 15px 0 5px 0;">⚠️ 需要改善</div>';

                rankings.worst.forEach((item, i) => {{
                    html += `
                        <div class="ranking-item worst">
                            <span class="rank-medal">${{medals[i]}}</span>
                            <span class="rank-province">${{item[0]}}</span>
                            <span class="rank-value">${{item[1].toFixed(1)}}</span>
                        </div>
                    `;
                }});

                document.getElementById('rankingContent').innerHTML = html;
            }}

            // 捕获Leaflet地图实例的函数
            function captureLeafletMap() {{
                console.log('尝试捕获Leaflet地图实例...');

                // 方法0: 检查是否已经暴露为mainLeafletMap
                if (window.mainLeafletMap) {{
                    window.leafletMap = window.mainLeafletMap;
                    console.log('✅ 方法0成功: 使用预暴露的 mainLeafletMap');
                    return true;
                }}

                // 方法1: 通过全局变量查找
                for (let key in window) {{
                    if (key.startsWith('map_') && window[key] instanceof L.Map) {{
                        window.leafletMap = window[key];
                        console.log('✅ 方法1成功: 通过全局变量 ' + key + ' 捕获到地图实例');
                        return true;
                    }}
                }}

                // 方法2: 通过DOM元素查找
                const mapDiv = document.getElementById('map');
                if (mapDiv) {{
                    const leafletContainer = mapDiv.querySelector('.leaflet-container');
                    if (leafletContainer && leafletContainer._leaflet_id) {{
                        // 尝试直接从window对象获取
                        for (let key in window) {{
                            if (window[key] instanceof L.Map) {{
                                window.leafletMap = window[key];
                                console.log('✅ 方法2成功: 找到地图实例');
                                return true;
                            }}
                        }}
                    }}
                }}

                // 方法3: 遍历所有可能的map对象
                const allDivs = document.querySelectorAll('.leaflet-container');
                if (allDivs.length > 0) {{
                    for (let key in window) {{
                        if (window[key] instanceof L.Map) {{
                            window.leafletMap = window[key];
                            console.log('✅ 方法3成功: 通过leaflet-container找到地图实例');
                            return true;
                        }}
                    }}
                }}

                console.error('❌ 所有方法都失败了，无法捕获Leaflet地图实例');
                console.log('调试信息: L对象存在?', typeof L !== 'undefined');
                console.log('调试信息: map div存在?', !!document.getElementById('map'));
                console.log('调试信息: mainLeafletMap存在?', !!window.mainLeafletMap);
                return false;
            }}

            // 为地图的path元素添加省份标识
            function tagProvincePolygons() {{
                console.log('🏷️ 开始标记省份多边形...');

                if (!window.leafletMap) {{
                    console.error('❌ 地图实例未找到，无法标记');
                    return false;
                }}

                let taggedCount = 0;
                const provinceNameMap = {json.dumps(PROVINCE_NAME_MAP)};
                const normalizedNames = {{}};

                // 创建标准化名称映射
                Object.entries(provinceNameMap).forEach(([en, cn]) => {{
                    // 移除常见后缀
                    const normalized = cn.replace('省', '').replace('市', '').replace('自治区', '');
                    normalizedNames[normalized] = normalized;
                }});

                // 遍历所有图层
                window.leafletMap.eachLayer(function(layer) {{
                    if (layer.feature && layer.feature.properties) {{
                        const provinceName = layer.feature.properties.name || '';
                        if (provinceName) {{
                            // 标准化省份名称
                            const normalized = provinceName.replace('省', '').replace('市', '').replace('自治区', '').replace('壮族自治区', '').replace('回族自治区', '').replace('维吾尔自治区', '').replace('特别行政区', '');

                            // 找到该图层对应的DOM元素
                            if (layer._path) {{
                                layer._path.dataset.province = normalized;
                                taggedCount++;
                                console.log('  ✅ 标记:', normalized);
                            }}
                        }}
                    }}
                }});

                console.log('✅ 标记完成:', taggedCount, '个省份');
                return taggedCount > 0;
            }}

            // 页面加载完成后初始化
            window.addEventListener('load', function() {{
                console.log('📄 页面加载完成，开始初始化...');

                // 初始化第一个图层的颜色渐变
                const firstMetadata = layerMetadata[0];
                console.log('第一个图层元数据:', firstMetadata);

                if (firstMetadata.colors && firstMetadata.colors.length > 0) {{
                    const gradient = 'linear-gradient(to right, ' + firstMetadata.colors.join(', ') + ')';
                    console.log('初始颜色渐变:', gradient.substring(0, 80) + '...');
                    const gradientElement = document.getElementById('legendGradient');
                    if (gradientElement) {{
                        gradientElement.style.background = gradient;
                        console.log('✅ 初始图例颜色已设置');
                    }} else {{
                        console.error('❌ 初始化时找不到legendGradient元素！');
                    }}
                }} else {{
                    console.error('❌ 第一个图层没有颜色数据！');
                }}

                // 捕获Leaflet地图实例并标记省份 - 多次尝试
                let attempts = 0;
                const maxAttempts = 10;
                const tryInitialize = function() {{
                    attempts++;
                    console.log('第', attempts, '次尝试捕获地图实例和标记省份...');

                    if (captureLeafletMap()) {{
                        console.log('🎉 地图实例捕获成功！');

                        // 标记省份多边形
                        setTimeout(function() {{
                            if (tagProvincePolygons()) {{
                                console.log('🎊 所有初始化完成！地图已准备就绪。');
                            }} else {{
                                console.warn('⚠️ 标记省份失败，将在下次尝试');
                                if (attempts < maxAttempts) {{
                                    setTimeout(tryInitialize, 500);
                                }}
                            }}
                        }}, 200);
                        return;
                    }}

                    if (attempts < maxAttempts) {{
                        setTimeout(tryInitialize, 300 * attempts); // 逐渐增加延迟
                    }} else {{
                        console.error('⚠️ 已尝试', maxAttempts, '次，仍无法完全初始化');
                        console.log('💡 图例颜色切换仍可工作，但地图着色切换可能无效');
                    }}
                }};

                setTimeout(tryInitialize, 300);
            }});
        </script>
    </body>
    """

    # 10. 保存地图
    map_path = os.path.join(OUTPUT_DIR, 'province_level_water_quality_map.html')
    m.save(map_path)

    # 11. 读取保存的HTML文件并注入自定义样式
    with open(map_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 将自定义HTML插入到body标签之后
    html_content = html_content.replace('<body>', custom_html)

    # 调整地图div的ID - 更精确的替换
    # 查找folium创建的div并添加id="map"
    import re
    # 匹配folium创建的主div容器
    html_content = re.sub(
        r'(<div[^>]*class="folium-map"[^>]*)',
        r'\1 id="map"',
        html_content
    )

    # 在</body>之前添加一个脚本来暴露地图实例
    expose_map_script = """
    <script>
        // 暴露地图实例到全局作用域
        (function() {
            // 等待Leaflet加载完成
            const exposeInterval = setInterval(function() {
                if (typeof L !== 'undefined') {
                    clearInterval(exposeInterval);
                    // 查找所有Leaflet地图实例并暴露到window
                    for (let key in window) {
                        if (key.match(/^map_[a-f0-9]+$/) && window[key] instanceof L.Map) {
                            window.mainLeafletMap = window[key];
                            console.log('✅ 地图实例已暴露为 window.mainLeafletMap');
                            break;
                        }
                    }
                }
            }, 100);
        })();
    </script>
    """
    html_content = html_content.replace('</body>', expose_map_script + '</body>')

    # 写回文件
    with open(map_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[OK] Modern map saved successfully to: {map_path}")
    print("[INFO] The map features a modern UI with enhanced interactivity and data insights.")

def main():
    """主函数，用于生成地图"""
    print("\n" + "="*80)
    print(" " * 10 + "WATER QUALITY MAP - MODERN UI GENERATION")
    print("="*80 + "\n")

    print(f"Fetching and processing GeoJSON data from: {GEOJSON_URL}")
    geo_data, province_centers = get_geojson_data(GEOJSON_URL)
    if not geo_data or not province_centers:
        print("Could not generate map due to missing GeoJSON data.")
        return
    print("[OK] GeoJSON data processed successfully.")

    INPUT_FILE = 'preprocessed_data_feature_eng.csv'
    print(f"Loading data from: {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"[OK] Data loaded successfully: {len(df)} records\n")

    create_province_choropleth_map(df, geo_data, province_centers)

    print("\n" + "="*80)
    print(" " * 15 + "MODERN MAP GENERATION COMPLETED!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
