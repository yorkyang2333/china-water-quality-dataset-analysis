"""
生成地理位置错误可视化地图（最终优化版）
展示数据集中经纬度与省份不对应的情况
突出显示位置错误的点位

策略：
- 位置"正确"的点（距离真实位置<100km）：稀疏采样10%
- 位置"错误"的点（距离真实位置≥100km）：全部显示，突出问题

输出：交互式HTML地图
作者：Claude Code
日期：2025-12-19
"""

import pandas as pd
import folium
import warnings
from math import radians, cos, sin, asin, sqrt
import numpy as np

warnings.filterwarnings('ignore')

# ==================== 常量定义 ====================

# 真实省份坐标（使用省会城市经纬度）
REAL_PROVINCE_COORDS = {
    'Beijing': (39.9042, 116.4074),      # 北京
    'Shanghai': (31.2304, 121.4737),     # 上海
    'Guangdong': (23.1291, 113.2644),    # 广州
    'Zhejiang': (30.2741, 120.1551),     # 杭州
    'Jiangsu': (32.0603, 118.7969),      # 南京
    'Shandong': (36.6512, 117.1201),     # 济南
    'Henan': (34.7466, 113.6253),        # 郑州
    'Hubei': (30.5928, 114.3055),        # 武汉
    'Sichuan': (30.5728, 104.0668),      # 成都
    'Yunnan': (25.0406, 102.7123),       # 昆明
}

# 中文名称映射
PROVINCE_CN_NAMES = {
    'Beijing': '北京', 'Shanghai': '上海', 'Guangdong': '广东',
    'Zhejiang': '浙江', 'Jiangsu': '江苏', 'Shandong': '山东',
    'Henan': '河南', 'Hubei': '湖北', 'Sichuan': '四川', 'Yunnan': '云南'
}

# 为每个省份分配不同颜色
PROVINCE_COLORS = {
    'Beijing': '#e74c3c',       # 红色
    'Shanghai': '#3498db',      # 蓝色
    'Guangdong': '#2ecc71',     # 绿色
    'Zhejiang': '#f39c12',      # 橙色
    'Jiangsu': '#9b59b6',       # 紫色
    'Shandong': '#1abc9c',      # 青色
    'Henan': '#e67e22',         # 深橙色
    'Hubei': '#34495e',         # 深灰色
    'Sichuan': '#95a5a6',       # 灰色
    'Yunnan': '#16a085',        # 深青���
}

# 距离阈值（公里）
DISTANCE_THRESHOLD = 100  # 超过此距离视为"位置错误"


def haversine(lon1, lat1, lon2, lat2):
    """计算两点间的球面距离（公里）"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def load_data():
    """加载预处理后的数据"""
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')

    print(f"[OK] 加载数据: {len(df)} 条记录")
    print(f"[OK] 省份数: {df['Province'].nunique()}")

    return df


def filter_data_points(df):
    """
    过滤数据点：
    - 位置正确的点（<100km）：稀疏采样1%（极少展示，避免干扰）
    - 位置错误的点（≥100km）：采样22%（充分展示问题，总量约20%）
    """
    print("\n[INFO] 过滤数据点...")

    filtered_data = []
    stats = {
        'total': len(df),
        'correct': 0,
        'correct_sampled': 0,
        'wrong': 0,
        'wrong_sampled': 0
    }

    for province in PROVINCE_COLORS.keys():
        province_data = df[df['Province'] == province].copy()
        if len(province_data) == 0:
            continue

        real_lat, real_lon = REAL_PROVINCE_COORDS[province]

        # 计算每个点与真实位置的距离
        province_data['distance'] = province_data.apply(
            lambda row: haversine(real_lon, real_lat, row['Longitude'], row['Latitude']),
            axis=1
        )

        # 分为"位置正确"和"位置错误"两组
        correct_points = province_data[province_data['distance'] < DISTANCE_THRESHOLD]
        wrong_points = province_data[province_data['distance'] >= DISTANCE_THRESHOLD]

        stats['correct'] += len(correct_points)
        stats['wrong'] += len(wrong_points)

        # 对"位置正确"的点进行极稀疏采样（1%）
        if len(correct_points) > 0:
            sample_size = max(1, int(len(correct_points) * 0.01))
            correct_sampled = correct_points.sample(n=sample_size, random_state=42)
            stats['correct_sampled'] += len(correct_sampled)
            filtered_data.append(correct_sampled)

        # "位置错误"的点采样22%（总量约20%）
        if len(wrong_points) > 0:
            sample_size = max(10, int(len(wrong_points) * 0.22))  # 至少10个点
            wrong_sampled = wrong_points.sample(n=min(sample_size, len(wrong_points)), random_state=42)
            stats['wrong_sampled'] += len(wrong_sampled)
            filtered_data.append(wrong_sampled)

    result_df = pd.concat(filtered_data, ignore_index=True)

    print(f"[OK] 原始数据点: {stats['total']}")
    print(f"[OK] 位置正确点(<{DISTANCE_THRESHOLD}km): {stats['correct']} -> 采样后: {stats['correct_sampled']}")
    print(f"[OK] 位置错误点(≥{DISTANCE_THRESHOLD}km): {stats['wrong']} -> 采样后: {stats['wrong_sampled']}")
    print(f"[OK] 最终显示点数: {len(result_df)} (原始数据的 {len(result_df)/stats['total']*100:.1f}%)")

    return result_df, stats


def create_interactive_map(df, stats):
    """创建交互式地图（无聚类，突出错误点位）"""
    print("\n[INFO] 创建交互式地图...")

    # 创建地图，中心设置为中国中心位置
    m = folium.Map(
        location=[35.0, 105.0],
        zoom_start=5,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # 1. 创建真实省份位置图层
    real_province_layer = folium.FeatureGroup(name='✓ 真实省份位置', show=True)

    for province, (lat, lon) in REAL_PROVINCE_COORDS.items():
        cn_name = PROVINCE_CN_NAMES[province]

        # 添加带省份名称的圆点标记（缩小尺寸）
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'''<div style="
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    background-color: green;
                    border: 2px solid darkgreen;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 9pt;
                    font-weight: bold;
                    color: white;
                    text-shadow: 1px 1px 2px black;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
                ">{cn_name}</div>''',
                icon_size=(35, 35),
                icon_anchor=(17, 17)
            ),
            popup=folium.Popup(f'<b>{cn_name}</b><br>真实位置<br>纬度: {lat:.4f}<br>经度: {lon:.4f}', max_width=200),
            tooltip=f'{cn_name}（真实位置）'
        ).add_to(real_province_layer)

    real_province_layer.add_to(m)

    # 2. 为每个省份创建数据点图层
    for province in PROVINCE_COLORS.keys():
        province_data = df[df['Province'] == province]
        if len(province_data) == 0:
            continue

        cn_name = PROVINCE_CN_NAMES[province]
        color = PROVINCE_COLORS[province]

        # 创建该省份的图层
        layer = folium.FeatureGroup(name=f'{cn_name} ({len(province_data)}个)', show=True)

        # 添加每个数据点
        for idx, row in province_data.iterrows():
            real_lat, real_lon = REAL_PROVINCE_COORDS[province]
            distance = row['distance']

            # 根据距离判断点的大小和透明度
            is_wrong = distance >= DISTANCE_THRESHOLD
            radius = 6 if is_wrong else 4
            opacity = 0.8 if is_wrong else 0.5
            weight = 2 if is_wrong else 1

            # 弹出信息
            popup_html = f"""
            <div style="font-family: Arial; font-size: 10pt;">
                <b>{cn_name}</b> - {row['City']}<br>
                <b>监测站:</b> {row['Monitoring_Station']}<br>
                <b>数据集坐标:</b><br>
                &nbsp;&nbsp;纬度: {row['Latitude']:.4f}<br>
                &nbsp;&nbsp;经度: {row['Longitude']:.4f}<br>
                <b>真实坐标:</b><br>
                &nbsp;&nbsp;纬度: {real_lat:.4f}<br>
                &nbsp;&nbsp;经度: {real_lon:.4f}<br>
                <b style="color: {'red' if is_wrong else 'orange'};">偏差距离: {distance:.1f} 公里</b><br>
                <b>采样日期:</b> {row['Date']}<br>
                <b>WQI:</b> {row['Water_Quality_Index']:.2f}<br>
                <b>污染等级:</b> {row['Pollution_Level']}
            </div>
            """

            # 添加圆点标记
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{cn_name} - {row['City']} (偏差: {distance:.0f}km)",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=opacity,
                weight=weight
            ).add_to(layer)

        layer.add_to(m)

    # 3. 创建连线图层（展示偏差）
    line_layer = folium.FeatureGroup(name='偏差连线示例', show=False)

    # 每个省份随机选取一些错误点画连线
    for province in PROVINCE_COLORS.keys():
        province_data = df[df['Province'] == province]
        # 只选择位置错误的点
        wrong_data = province_data[province_data['distance'] >= DISTANCE_THRESHOLD]

        if len(wrong_data) == 0:
            continue

        sample = wrong_data.sample(n=min(5, len(wrong_data)), random_state=42)
        real_lat, real_lon = REAL_PROVINCE_COORDS[province]
        color = PROVINCE_COLORS[province]

        for idx, row in sample.iterrows():
            folium.PolyLine(
                locations=[[real_lat, real_lon], [row['Latitude'], row['Longitude']]],
                color=color,
                weight=2.5,
                opacity=0.8,
                dash_array='8, 4'
            ).add_to(line_layer)

    line_layer.add_to(m)

    # 添加图层控制器
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    # 添加颜色图例
    legend_html = '''
    <div style="position: fixed;
                top: 10px;
                left: 50px;
                width: 420px;
                background-color: white;
                border: 2px solid grey;
                z-index: 9999;
                font-size: 12px;
                padding: 12px;
                border-radius: 5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; font-size: 14px;">数据集地理位置错误可视化</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 10px;">
    '''

    # 添加省份颜色图例
    for province, color in PROVINCE_COLORS.items():
        cn_name = PROVINCE_CN_NAMES[province]
        legend_html += f'<div><span style="color: {color}; font-size: 14px;">●</span> {cn_name}</div>'

    legend_html += f'''
        </div>
        <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ccc;">
        <div style="font-size: 10px; line-height: 1.6;">
            <span style="color: green; font-weight: bold;">● 绿色</span>：真实省份位置<br>
            <span style="font-weight: bold;">● 彩色大点</span>：位置错误(≥{DISTANCE_THRESHOLD}km)<br>
            <span style="font-weight: bold; opacity: 0.5;">● 彩色小点</span>：位置正确(&lt;{DISTANCE_THRESHOLD}km)
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 添加统计信息
    display_total = stats['wrong_sampled'] + stats['correct_sampled']
    wrong_pct = stats['wrong'] / stats['total'] * 100

    stats_html = f'''
    <div style="position: fixed;
                bottom: 10px;
                left: 10px;
                width: 250px;
                background-color: white;
                border: 2px solid grey;
                z-index: 9999;
                font-size: 11px;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 8px 0; font-size: 13px;">数据统计</h4>
        <p style="margin: 3px 0; font-size: 10px; line-height: 1.6;">
            <b>原始数据:</b> {stats['total']} 个点<br>
            <b style="color: red;">位置错误:</b> {stats['wrong']} ({wrong_pct:.1f}%)<br>
            <b>位置正确:</b> {stats['correct']} ({100-wrong_pct:.1f}%)<br>
        </p>
        <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ccc;">
        <p style="margin: 6px 0 0 0; font-size: 10px; line-height: 1.6;">
            <b>地图显示:</b> {display_total} 个点<br>
            &nbsp;&nbsp;错误点: {stats['wrong_sampled']} (采样22%)<br>
            &nbsp;&nbsp;正确点: {stats['correct_sampled']} (采样1%)
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    return m


def main():
    """主函数"""
    print("=" * 65)
    print("          地理位置错误可视化地图生成器（最终版）")
    print("=" * 65)
    print()

    # 1. 加载数据
    df = load_data()

    # 2. 过滤数据点
    filtered_df, stats = filter_data_points(df)

    # 3. 创建地图
    m = create_interactive_map(filtered_df, stats)

    # 4. 保存地图
    output_file = 'geo_error_map.html'
    m.save(output_file)
    print(f"\n[DONE] 地图已保存: {output_file}")

    print("\n" + "=" * 65)
    print("完成！")
    print("=" * 65)
    print(f"\n请在浏览器中打开 {output_file} 查看交互式地图。")
    print("\n设计理念：")
    print(f"  - 位置错误点(≥{DISTANCE_THRESHOLD}km)：采样22%，大且明显（突出问题）")
    print(f"  - 位置相对正确点(<{DISTANCE_THRESHOLD}km)：采样1%（极少展示，避免干扰）")
    print("\n可视化特点：")
    print("  - 真实省份位置：绿色圆点（内含省份名称）")
    print("  - 数据点：按省份着色，位置错误的点更大更明显")
    print("  - 偏差连线：可选显示，连接真实位置与错误位置")
    print("  - 地图清爽，问题突出，适合PPT展示")
    print()


if __name__ == '__main__':
    main()
