"""
数据服务模块
提供数据查询、过滤、统计等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class DataService:
    def __init__(self, data_path):
        """初始化数据服务"""
        print(f"Loading data from: {data_path}")
        try:
            self.df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(data_path, encoding='utf-8-sig')

        # 确保日期列为datetime类型
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        elif 'Sampling_Date' in self.df.columns:
            self.df['Sampling_Date'] = pd.to_datetime(self.df['Sampling_Date'])
            self.df.rename(columns={'Sampling_Date': 'Date'}, inplace=True)

        # 加载站点位置信息（生成模拟坐标）
        self._generate_station_coordinates()

        print(f"Data loaded successfully: {len(self.df)} records")

    def _generate_station_coordinates(self):
        """为每个监测站生成坐标（基于省份中心点+随机偏移）"""
        # 省份中心坐标
        province_centers = {
            'Beijing': [39.9042, 116.4074],
            'Shanghai': [31.2304, 121.4737],
            'Guangdong': [23.1291, 113.2644],
            'Jiangsu': [32.0603, 118.7969],
            'Zhejiang': [30.2741, 120.1551],
            'Shandong': [36.6512, 117.1201],
            'Henan': [34.7466, 113.6254],
            'Sichuan': [30.5728, 104.0668],
            'Hubei': [30.5928, 114.3055],
            'Yunnan': [25.0406, 102.7103]
        }

        # 为每个站点生成唯一坐标
        station_coords = {}
        np.random.seed(42)  # 确保坐标可重复

        for province, stations in self.df.groupby('Province')['Monitoring_Station'].unique().items():
            center = province_centers.get(province, [35.0, 105.0])
            for i, station in enumerate(stations):
                # 在省份中心周围随机分布
                lat_offset = np.random.uniform(-1.5, 1.5)
                lng_offset = np.random.uniform(-1.5, 1.5)
                station_coords[station] = [
                    center[0] + lat_offset,
                    center[1] + lng_offset
                ]

        self.station_coords = station_coords

    def get_all_stations(self):
        """获取所有监测站点信息"""
        stations_info = []

        for station_name, coords in self.station_coords.items():
            station_data = self.df[self.df['Monitoring_Station'] == station_name]

            if len(station_data) > 0:
                latest_record = station_data.sort_values('Date', ascending=False).iloc[0]

                stations_info.append({
                    'name': station_name,
                    'province': latest_record['Province'],
                    'coordinates': coords,
                    'latest_wqi': float(latest_record['Water_Quality_Index']),
                    'pollution_level': latest_record['Pollution_Level'],
                    'record_count': int(len(station_data)),
                    'latest_date': latest_record['Date'].strftime('%Y-%m-%d') if pd.notna(latest_record['Date']) else None
                })

        return stations_info

    def get_timeline_data(self, start_date=None, end_date=None, station=None, province=None):
        """获取时间轴数据"""
        df_filtered = self.df.copy()

        # 日期过滤
        if start_date:
            df_filtered = df_filtered[df_filtered['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_filtered = df_filtered[df_filtered['Date'] <= pd.to_datetime(end_date)]

        # 站点/省份过滤
        if station:
            df_filtered = df_filtered[df_filtered['Monitoring_Station'] == station]
        if province:
            df_filtered = df_filtered[df_filtered['Province'] == province]

        # 按日期聚合
        timeline = df_filtered.groupby('Date').agg({
            'Water_Quality_Index': 'mean',
            'Pollution_Level': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
            'Monitoring_Station': 'count'
        }).reset_index()

        timeline['Date'] = timeline['Date'].dt.strftime('%Y-%m-%d')
        timeline.rename(columns={'Date': 'Sampling_Date'}, inplace=True)

        return timeline.to_dict('records')

    def get_spatial_data(self, query):
        """空间查询数据"""
        query_type = query.get('type')
        df_filtered = self.df.copy()

        # 日期过滤
        if query.get('start_date'):
            df_filtered = df_filtered[df_filtered['Date'] >= pd.to_datetime(query['start_date'])]
        if query.get('end_date'):
            df_filtered = df_filtered[df_filtered['Date'] <= pd.to_datetime(query['end_date'])]

        # 空间过滤
        if query_type == 'rectangle':
            bounds = query.get('bounds', [])
            if len(bounds) == 2:
                lat_min = min(bounds[0][0], bounds[1][0])
                lat_max = max(bounds[0][0], bounds[1][0])
                lng_min = min(bounds[0][1], bounds[1][1])
                lng_max = max(bounds[0][1], bounds[1][1])

                # 过滤站点
                selected_stations = []
                for station, coords in self.station_coords.items():
                    if lat_min <= coords[0] <= lat_max and lng_min <= coords[1] <= lng_max:
                        selected_stations.append(station)

                df_filtered = df_filtered[df_filtered['Monitoring_Station'].isin(selected_stations)]

        elif query_type == 'circle':
            center = query.get('center', [])
            radius_km = query.get('radius', 50) / 1000  # 转换为公里

            if len(center) == 2:
                # 简化计算：使用欧氏距离近似
                selected_stations = []
                for station, coords in self.station_coords.items():
                    # 大约111km每度
                    dist = np.sqrt(
                        ((coords[0] - center[0]) * 111) ** 2 +
                        ((coords[1] - center[1]) * 111 * np.cos(np.radians(center[0]))) ** 2
                    )
                    if dist <= radius_km:
                        selected_stations.append(station)

                df_filtered = df_filtered[df_filtered['Monitoring_Station'].isin(selected_stations)]

        # 统计结果
        if len(df_filtered) == 0:
            return {
                'record_count': 0,
                'stations': [],
                'statistics': {}
            }

        # 计算统计信息
        statistics = {
            'avg_wqi': float(df_filtered['Water_Quality_Index'].mean()),
            'avg_cod': float(df_filtered['COD_mg_L'].mean()),
            'avg_ph': float(df_filtered['pH'].mean()),
            'pollution_distribution': df_filtered['Pollution_Level'].value_counts().to_dict(),
            'station_count': int(df_filtered['Monitoring_Station'].nunique())
        }

        # 站点详情
        stations = []
        for station in df_filtered['Monitoring_Station'].unique():
            station_data = df_filtered[df_filtered['Monitoring_Station'] == station]
            stations.append({
                'name': station,
                'coordinates': self.station_coords.get(station, [0, 0]),
                'avg_wqi': float(station_data['Water_Quality_Index'].mean()),
                'record_count': int(len(station_data))
            })

        return {
            'record_count': int(len(df_filtered)),
            'stations': stations,
            'statistics': statistics
        }

    def get_statistics_summary(self, province=None):
        """获取统计摘要"""
        df_filtered = self.df if province is None else self.df[self.df['Province'] == province]

        summary = {
            'total_records': int(len(df_filtered)),
            'total_stations': int(df_filtered['Monitoring_Station'].nunique()),
            'avg_wqi': float(df_filtered['Water_Quality_Index'].mean()),
            'pollution_distribution': df_filtered['Pollution_Level'].value_counts().to_dict(),
            'provinces': int(df_filtered['Province'].nunique()) if province is None else 1,
            'date_range': {
                'start': df_filtered['Date'].min().strftime('%Y-%m-%d') if pd.notna(df_filtered['Date'].min()) else None,
                'end': df_filtered['Date'].max().strftime('%Y-%m-%d') if pd.notna(df_filtered['Date'].max()) else None
            },
            'indicators': {
                'wqi': {
                    'mean': float(df_filtered['Water_Quality_Index'].mean()),
                    'min': float(df_filtered['Water_Quality_Index'].min()),
                    'max': float(df_filtered['Water_Quality_Index'].max())
                },
                'cod': {
                    'mean': float(df_filtered['COD_mg_L'].mean()),
                    'min': float(df_filtered['COD_mg_L'].min()),
                    'max': float(df_filtered['COD_mg_L'].max())
                },
                'ph': {
                    'mean': float(df_filtered['pH'].mean()),
                    'min': float(df_filtered['pH'].min()),
                    'max': float(df_filtered['pH'].max())
                }
            }
        }

        return summary

    def get_trends(self, indicator='Water_Quality_Index', province=None, station=None, interval='month'):
        """获取趋势数据"""
        df_filtered = self.df.copy()

        if province:
            df_filtered = df_filtered[df_filtered['Province'] == province]
        if station:
            df_filtered = df_filtered[df_filtered['Monitoring_Station'] == station]

        # 按时间间隔聚合
        if interval == 'day':
            df_filtered['period'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')
        elif interval == 'week':
            df_filtered['period'] = df_filtered['Date'].dt.strftime('%Y-W%U')
        else:  # month
            df_filtered['period'] = df_filtered['Date'].dt.strftime('%Y-%m')

        trends = df_filtered.groupby('period')[indicator].mean().reset_index()
        trends.columns = ['period', 'value']

        return trends.to_dict('records')
