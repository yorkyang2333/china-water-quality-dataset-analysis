import pandas as pd
import os
import folium
from folium import FeatureGroup
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
    """使用Folium创建按省份聚合的中国水质地图，包含多个可切换图层"""
    print("CREATING PROVINCE-LEVEL WATER QUALITY MAP WITH MULTIPLE LAYERS")

    # 1. 定义要可视化的指标及其配置
    indicators_to_visualize = {
        'Water_Quality_Index': {'name': 'Avg. Water Quality Index', 'color': 'YlOrRd'},
        'Total_Heavy_Metals_ug_L': {'name': 'Avg. Total Heavy Metals (µg/L)', 'color': 'Reds'},
        'COD_mg_L': {'name': 'Avg. COD (mg/L)', 'color': 'Oranges'},
        'Total_Nitrogen_mg_L': {'name': 'Avg. Total Nitrogen (mg/L)', 'color': 'Purples'},
        'Total_Phosphorus_mg_L': {'name': 'Avg. Total Phosphorus (mg/L)', 'color': 'BuPu'},
        'Dissolved_Oxygen_mg_L': {'name': 'Avg. Dissolved Oxygen (mg/L)', 'color': 'GnBu_r'}
    }
    
    # 2. 按省份聚合所有需要的指标
    agg_dict = {indicator: 'mean' for indicator in indicators_to_visualize}
    agg_dict['Pollution_Level'] = lambda x: x.value_counts().index[0]
    
    province_agg = df.groupby('Province').agg(agg_dict).reset_index()
    province_agg.rename(columns={'Pollution_Level': 'Most_Common_Pollution_Level'}, inplace=True)

    # 3. 添加中文名称和标准化中文名称列
    province_agg['Province_CN'] = province_agg['Province'].map(PROVINCE_NAME_MAP)
    province_agg['Province_CN_Normalized'] = province_agg['Province_CN'].apply(normalize_province_name)

    # 4. 创建底图
    map_center = [35.8617, 104.1954]
    m = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB positron')

    # 5. 循环创建多个Choropleth图层
    print("\nCreating map layers...")
    for i, (indicator, config) in enumerate(indicators_to_visualize.items()):
        is_visible = (i == 0)
        layer_name = config['name']
        
        folium.Choropleth(
            geo_data=geo_data,
            name=layer_name,
            data=province_agg,
            columns=['Province_CN_Normalized', indicator], # 使用标准化的中文名进行匹配
            key_on='feature.properties.normalized_name', # 使用GeoJSON中新增的标准化名称字段
            fill_color=config['color'],
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=layer_name,
            highlight=True,
            nan_fill_color='gray',
            show=is_visible
        ).add_to(m)
        print(f" - Layer '{layer_name}' created.")

    # 6. 创建并添加省份信息标记图层
    marker_group = FeatureGroup(name='Province Info Markers', show=True)
    for _, row in province_agg.iterrows():
        if pd.isna(row['Province_CN_Normalized']):
            continue
        center_coords = province_centers.get(row['Province_CN_Normalized'])
        if center_coords:
            popup_html = f"<h4><b>{row['Province']} ({row['Province_CN']})</b></h4><hr>"
            popup_html += f"<b>Most Common Pollution Level:</b> {row['Most_Common_Pollution_Level']}<br><br>"
            for indicator, config in indicators_to_visualize.items():
                popup_html += f"<b>{config['name']}:</b> {row[indicator]:.2f}<br>"
            popup = folium.Popup(popup_html, max_width=350)
            folium.Marker(
                location=center_coords,
                popup=popup,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_group)
    marker_group.add_to(m)

    # 7. 添加图层控制器
    folium.LayerControl(collapsed=False).add_to(m)

    # 8. 保存地图
    map_path = os.path.join(OUTPUT_DIR, 'province_level_water_quality_map.html')
    m.save(map_path)
    print(f"\nMap saved successfully to: {map_path}")
    print("The map is interactive. Open the HTML file in a browser to explore the layers.")

def main():
    """主函数，用于生成地图"""
    print("\n" + "="*80)
    print(" " * 10 + "WATER POLLUTION EDA - MULTI-LAYER PROVINCE MAP GENERATION")
    print("="*80 + "\n")

    print(f"Fetching and processing GeoJSON data from: {GEOJSON_URL}")
    geo_data, province_centers = get_geojson_data(GEOJSON_URL)
    if not geo_data or not province_centers:
        print("Could not generate map due to missing GeoJSON data.")
        return
    print("GeoJSON data processed successfully.")
    
    INPUT_FILE = 'preprocessed_data_feature_eng.csv'
    print(f"Loading data from: {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"Data loaded successfully: {len(df)} records\n")

    create_province_choropleth_map(df, geo_data, province_centers)

    print("\n" + "="*80)
    print(" " * 20 + "MAP GENERATION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
