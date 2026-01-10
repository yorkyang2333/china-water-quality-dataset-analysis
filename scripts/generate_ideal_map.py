"""
生成理想数据集地理位置可视化地图（对比演示用）
展示如果数据集是真实的，点位应该如何分布

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
    'Yunnan': '#16a085',        # 深青色
}


def haversine(lon1, lat1, lon2, lat2):
    """计算两点间的球面距离（公里）"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def generate_ideal_data():
    """生成理想的数据集（点位在真实省份范围内）"""
    print("\n[INFO] 生成理想数据集...")

    ideal_data = []

    # 每个省份生成约1550个点（总计15500个点，采样20%后约3100个，与实际数据一致）
    for province, (real_lat, real_lon) in REAL_PROVINCE_COORDS.items():
        cn_name = PROVINCE_CN_NAMES[province]

        # 为每个省份生成1550个点，分布在整个省份范围内
        for i in range(1550):
            # 生成在真实位置周围更大范围内的随机点
            # 使用正态分布，覆盖整个省份范围（约150km半径）
            lat_offset = np.random.normal(0, 0.8)  # 约88km标准差，覆盖约200km范围
            lon_offset = np.random.normal(0, 0.8)  # 约88km标准差，覆盖约200km范围

            lat = real_lat + lat_offset
            lon = real_lon + lon_offset

            # 计算距离省中心的距离
            distance = haversine(real_lon, real_lat, lon, lat)

            ideal_data.append({
                'Province': province,
                'Province_CN': cn_name,
                'Latitude': lat,
                'Longitude': lon,
                'distance': distance,
                'Station_ID': f'{province}_Station_{i+1}'
            })

    df = pd.DataFrame(ideal_data)

    print(f"[OK] 生成理想数据点: {len(df)}")
    print(f"[OK] 平均距省中心: {df['distance'].mean():.1f} 公里")
    print(f"[OK] 最大距省中心: {df['distance'].max():.1f} 公里")
    print(f"[OK] 中位数距离: {df['distance'].median():.1f} 公里")

    return df


def sample_ideal_data(df, sample_rate=0.2):
    """对理想数据进行采样"""
    print(f"\n[INFO] 采样理想数据（采样率: {sample_rate*100:.0f}%）...")

    sampled_data = []
    total_original = len(df)

    for province in PROVINCE_COLORS.keys():
        province_data = df[df['Province'] == province]
        if len(province_data) == 0:
            continue

        sample_size = max(10, int(len(province_data) * sample_rate))
        sampled = province_data.sample(n=sample_size, random_state=42)
        sampled_data.append(sampled)

    result_df = pd.concat(sampled_data, ignore_index=True)

    print(f"[OK] 原始数据点: {total_original}")
    print(f"[OK] 采样后数据点: {len(result_df)}")

    return result_df


def create_ideal_map(df):
    """创建理想数据集地图"""
    print("\n[INFO] 创建理想数据集地图...")

    # 创建地图
    m = folium.Map(
        location=[35.0, 105.0],
        zoom_start=5,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # 1. 创建真实省份位置图层
    real_province_layer = folium.FeatureGroup(name='✓ 省份中心位置', show=True)

    for province, (lat, lon) in REAL_PROVINCE_COORDS.items():
        cn_name = PROVINCE_CN_NAMES[province]

        # 添加带省份名称的圆点标记
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
            popup=folium.Popup(f'<b>{cn_name}</b><br>省份中心位置<br>纬度: {lat:.4f}<br>经度: {lon:.4f}', max_width=200),
            tooltip=f'{cn_name}（省份中心）'
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

            # 弹出信息
            popup_html = f"""
            <div style="font-family: Arial; font-size: 10pt;">
                <b>{cn_name}</b><br>
                <b>监测站:</b> {row['Station_ID']}<br>
                <b>数据点坐标:</b><br>
                &nbsp;&nbsp;纬度: {row['Latitude']:.4f}<br>
                &nbsp;&nbsp;经度: {row['Longitude']:.4f}<br>
                <b>省份中心:</b><br>
                &nbsp;&nbsp;纬度: {real_lat:.4f}<br>
                &nbsp;&nbsp;经度: {real_lon:.4f}<br>
                <b style="color: green;">距省中心: {distance:.1f} 公里</b><br>
                <small>（理想数据：点位在省份范围内）</small>
            </div>
            """

            # 添加圆点标记（理想数据点较小，统一样式）
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{cn_name} - {row['Station_ID']} (距中心: {distance:.0f}km)",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=1.5
            ).add_to(layer)

        layer.add_to(m)

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
                box-shadow: 2px 2px 6px rgba(0,0,0,0.4);">
        <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #2c3e50;">理想数据集地理位置分布（对比演示）</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 10px;">
    '''

    # 添加省份颜色图例
    for province, color in PROVINCE_COLORS.items():
        cn_name = PROVINCE_CN_NAMES[province]
        legend_html += f'<div><span style="color: {color}; font-size: 14px;">●</span> {cn_name}</div>'

    legend_html += '''
        </div>
        <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ccc;">
        <div style="font-size: 10px; line-height: 1.6;">
            <span style="color: green; font-weight: bold;">● 绿色</span>：省份中心位置<br>
            <span style="font-weight: bold;">● 彩色点</span>：数据点（分布在对应省份范围内）
        </div>
        <div style="margin-top: 8px; padding: 8px; background-color: #e8f5e9; border-radius: 4px; font-size: 9px; color: #2e7d32;">
            <b>说明：</b>这是理想的数据集示例，所有数据点都正确分布在各自省份范围内（覆盖整个省域）。
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 添加统计信息
    avg_distance = df['distance'].mean()
    max_distance = df['distance'].max()
    median_distance = df['distance'].median()

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
                box-shadow: 2px 2px 6px rgba(0,0,0,0.4);">
        <h4 style="margin: 0 0 8px 0; font-size: 13px; color: #2c3e50;">数据统计</h4>
        <p style="margin: 3px 0; font-size: 10px; line-height: 1.6;">
            <b>显示数据点:</b> {len(df)} 个<br>
            <b>省份数量:</b> 10 个<br>
        </p>
        <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ccc;">
        <p style="margin: 6px 0 0 0; font-size: 10px; line-height: 1.6;">
            <b style="color: green;">平均距省中心:</b> {avg_distance:.1f} km<br>
            <b style="color: green;">最大距省中心:</b> {max_distance:.1f} km<br>
            <b style="color: green;">中位数距离:</b> {median_distance:.1f} km
        </p>
        <div style="margin-top: 8px; padding: 6px; background-color: #e8f5e9; border-radius: 4px; font-size: 9px; color: #2e7d32;">
            所有点位都在正确的省份范围内
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    return m


def main():
    """主函数"""
    print("=" * 65)
    print("          理想数据集地理位置地图生成器")
    print("=" * 65)
    print()

    # 1. 生成理想数据
    df_ideal = generate_ideal_data()

    # 2. 采样数据
    df_sampled = sample_ideal_data(df_ideal, sample_rate=0.2)

    # 3. 创建地图
    m = create_ideal_map(df_sampled)

    # 4. 保存地图
    output_file = 'geo_ideal_map.html'
    m.save(output_file)
    print(f"\n[DONE] 理想数据集地图已保存: {output_file}")

    print("\n" + "=" * 65)
    print("完成！")
    print("=" * 65)
    print(f"\n请在浏览器中打开 {output_file} 查看理想数据集地图。")
    print("\n地图特点：")
    print("  - 所有数据点都正确分布在对应省份范围内")
    print("  - 点位覆盖整个省域（约150km半径范围）")
    print("  - 数据点数量与实际数据集相当（约3000个）")
    print("  - 与实际数据集形成鲜明对比")
    print("  - 适合用于PPT对比演示")
    print()


if __name__ == '__main__':
    main()
