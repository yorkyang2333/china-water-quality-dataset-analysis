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

        print(f"Columns in dataset: {list(self.df.columns)}")

        # 确保日期列为datetime类型
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            print("Using 'Date' column")
        elif 'Sampling_Date' in self.df.columns:
            self.df['Sampling_Date'] = pd.to_datetime(self.df['Sampling_Date'])
            self.df.rename(columns={'Sampling_Date': 'Date'}, inplace=True)
            print("Using 'Sampling_Date' column, renamed to 'Date'")
        else:
            print("WARNING: No Date or Sampling_Date column found!")

        # 打印日期范围
        if 'Date' in self.df.columns:
            print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")

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

        # 计算污染等级分布
        pollution_dist = df_filtered['Pollution_Level'].value_counts().to_dict()

        summary = {
            'total_records': int(len(df_filtered)),
            'total_stations': int(df_filtered['Monitoring_Station'].nunique()),
            'avg_wqi': float(df_filtered['Water_Quality_Index'].mean()),
            'pollution_distribution': pollution_dist,
            # 添加各等级的数量（用于前端显示）
            'excellent': int(pollution_dist.get('Excellent', 0)),
            'Excellent': int(pollution_dist.get('Excellent', 0)),
            'good': int(pollution_dist.get('Good', 0)),
            'Good': int(pollution_dist.get('Good', 0)),
            'fair': int(pollution_dist.get('Fair', 0)),
            'Fair': int(pollution_dist.get('Fair', 0)),
            'poor': int(pollution_dist.get('Poor', 0)),
            'Poor': int(pollution_dist.get('Poor', 0)),
            'very_poor': int(pollution_dist.get('Very Poor', 0)),
            'VeryPoor': int(pollution_dist.get('Very Poor', 0)),
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

    def get_date_range(self):
        """获取数据的日期范围"""
        if 'Date' not in self.df.columns:
            print("ERROR: Date column not found in dataframe")
            return {
                'min_date': None,
                'max_date': None
            }

        min_date = self.df['Date'].min()
        max_date = self.df['Date'].max()

        print(f"Date range requested: {min_date} to {max_date}")

        result = {
            'min_date': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None,
            'max_date': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None
        }

        print(f"Returning date range: {result}")
        return result

    def get_stations_by_date(self, date_str):
        """获取截止到指定日期的站点数据（显示每个站点在该日期前的最新记录）"""
        target_date = pd.to_datetime(date_str)

        # 过滤到目标日期之前的数据
        df_filtered = self.df[self.df['Date'] <= target_date]

        if len(df_filtered) == 0:
            return {
                'stations': [],
                'statistics': {
                    'total_stations': 0,
                    'total_records': 0,
                    'avg_wqi': 0
                }
            }

        # 为每个站点获取最新记录
        stations_info = []
        for station_name in df_filtered['Monitoring_Station'].unique():
            station_data = df_filtered[df_filtered['Monitoring_Station'] == station_name]
            latest_record = station_data.sort_values('Date', ascending=False).iloc[0]

            if station_name in self.station_coords:
                stations_info.append({
                    'name': station_name,
                    'province': latest_record['Province'],
                    'coordinates': self.station_coords[station_name],
                    'latest_wqi': float(latest_record['Water_Quality_Index']),
                    'pollution_level': latest_record['Pollution_Level'],
                    'record_count': int(len(station_data)),
                    'latest_date': latest_record['Date'].strftime('%Y-%m-%d') if pd.notna(latest_record['Date']) else None
                })

        # 计算统计信息
        statistics = {
            'total_stations': len(stations_info),
            'total_records': int(len(df_filtered)),
            'avg_wqi': float(df_filtered['Water_Quality_Index'].mean())
        }

        return {
            'stations': stations_info,
            'statistics': statistics
        }

    # ============= GIS检测功能方法 =============

    def get_realtime_monitoring(self, province=None):
        """获取实时监测指标"""
        df_filtered = self.df if province is None else self.df[self.df['Province'] == province]

        # 获取最新的数据记录
        latest_date = df_filtered['Date'].max()
        recent_data = df_filtered[df_filtered['Date'] >= (latest_date - timedelta(days=7))]

        # 定义标准阈值
        thresholds = {
            'Water_Quality_Index': {'good': 40, 'warning': 60, 'danger': 80},
            'COD_mg_L': {'good': 20, 'warning': 30, 'danger': 40},
            'pH': {'good_min': 6.5, 'good_max': 8.5, 'warning_min': 6.0, 'warning_max': 9.0},
            'Total_Nitrogen_mg_L': {'good': 1.0, 'warning': 2.0, 'danger': 3.0},
            'Total_Phosphorus_mg_L': {'good': 0.2, 'warning': 0.4, 'danger': 0.6},
            'Dissolved_Oxygen_mg_L': {'good': 6.0, 'warning': 4.0, 'danger': 2.0},
            'Ammonia_Nitrogen_mg_L': {'good': 1.0, 'warning': 1.5, 'danger': 2.0}
        }

        indicators = []

        for indicator_name, threshold in thresholds.items():
            if indicator_name not in df_filtered.columns:
                continue

            current_value = float(recent_data[indicator_name].mean())
            all_time_avg = float(df_filtered[indicator_name].mean())

            # 计算变化趋势
            if len(recent_data) > 1:
                recent_avg = recent_data.tail(3)[indicator_name].mean()
                prev_avg = recent_data.head(max(1, len(recent_data) - 3))[indicator_name].mean()
                trend = 'up' if recent_avg > prev_avg else 'down' if recent_avg < prev_avg else 'stable'
                change_percent = ((recent_avg - prev_avg) / prev_avg * 100) if prev_avg != 0 else 0
            else:
                trend = 'stable'
                change_percent = 0

            # 判断状态
            status = 'good'
            if indicator_name == 'pH':
                if threshold['good_min'] <= current_value <= threshold['good_max']:
                    status = 'good'
                elif threshold['warning_min'] <= current_value <= threshold['warning_max']:
                    status = 'warning'
                else:
                    status = 'danger'
            elif indicator_name == 'Dissolved_Oxygen_mg_L':
                # 溶解氧是越高越好
                if current_value >= threshold['good']:
                    status = 'good'
                elif current_value >= threshold['warning']:
                    status = 'warning'
                else:
                    status = 'danger'
            else:
                # 其他指标是越低越好
                if current_value <= threshold['good']:
                    status = 'good'
                elif current_value <= threshold['warning']:
                    status = 'warning'
                else:
                    status = 'danger'

            indicators.append({
                'name': indicator_name,
                'value': round(current_value, 2),
                'unit': self._get_unit(indicator_name),
                'status': status,
                'trend': trend,
                'change_percent': round(change_percent, 1),
                'all_time_avg': round(all_time_avg, 2)
            })

        return {
            'timestamp': latest_date.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(latest_date) else None,
            'indicators': indicators,
            'total_stations': int(df_filtered['Monitoring_Station'].nunique()),
            'records_count': int(len(recent_data))
        }

    def get_monitoring_alerts(self, province=None):
        """获取告警信息"""
        df_filtered = self.df if province is None else self.df[self.df['Province'] == province]

        # 获取最新数据 - 扩大时间范围到90天以确保有足够的告警数据
        latest_date = df_filtered['Date'].max()
        recent_data = df_filtered[df_filtered['Date'] >= (latest_date - timedelta(days=90))]

        alerts = []

        # 检查污染等级 - Poor和Very Poor
        pollution_alerts = recent_data[recent_data['Pollution_Level'].isin(['Poor', 'Very Poor'])]
        for _, row in pollution_alerts.head(50).iterrows():  # 增加到50条
            level = 'critical' if row['Pollution_Level'] == 'Very Poor' else 'warning'
            alerts.append({
                'type': 'pollution',
                'level': level,
                'station_name': row['Monitoring_Station'],
                'station': row['Monitoring_Station'],
                'province': row['Province'],
                'message': f"{row['Monitoring_Station']} 污染等级为 {row['Pollution_Level']}",
                'date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else None,
                'value': float(row['Water_Quality_Index'])
            })

        # 检查高水质指数（WQI > 80表示水质较差）
        high_wqi = recent_data[recent_data['Water_Quality_Index'] > 80]
        for _, row in high_wqi.head(30).iterrows():
            alerts.append({
                'type': 'indicator',
                'level': 'warning',
                'station_name': row['Monitoring_Station'],
                'station': row['Monitoring_Station'],
                'province': row['Province'],
                'message': f"{row['Monitoring_Station']} 水质指数偏高 (WQI: {row['Water_Quality_Index']:.1f})",
                'date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else None,
                'value': float(row['Water_Quality_Index'])
            })

        # 检查超标指标
        if 'COD_mg_L' in recent_data.columns:
            cod_alerts = recent_data[recent_data['COD_mg_L'] > 30]  # 降低阈值从40到30
            for _, row in cod_alerts.head(20).iterrows():
                alerts.append({
                    'type': 'indicator',
                    'level': 'critical',
                    'station_name': row['Monitoring_Station'],
                    'station': row['Monitoring_Station'],
                    'province': row['Province'],
                    'message': f"{row['Monitoring_Station']} COD超标 ({row['COD_mg_L']:.1f} mg/L)",
                    'date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else None,
                    'value': float(row['COD_mg_L'])
                })

        # 检查总氮超标
        if 'Total_Nitrogen_mg_L' in recent_data.columns:
            nitrogen_alerts = recent_data[recent_data['Total_Nitrogen_mg_L'] > 2.0]
            for _, row in nitrogen_alerts.head(20).iterrows():
                alerts.append({
                    'type': 'indicator',
                    'level': 'warning',
                    'station_name': row['Monitoring_Station'],
                    'station': row['Monitoring_Station'],
                    'province': row['Province'],
                    'message': f"{row['Monitoring_Station']} 总氮超标 ({row['Total_Nitrogen_mg_L']:.2f} mg/L)",
                    'date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else None,
                    'value': float(row['Total_Nitrogen_mg_L'])
                })

        # 去重（同一站点同一天只保留一条告警）
        seen = set()
        unique_alerts = []
        for alert in alerts:
            key = (alert['station_name'], alert['date'])
            if key not in seen:
                seen.add(key)
                unique_alerts.append(alert)

        # 按日期排序，最新的在前
        unique_alerts.sort(key=lambda x: x['date'] if x['date'] else '', reverse=True)

        return {
            'total_alerts': len(unique_alerts),
            'critical_count': len([a for a in unique_alerts if a['level'] == 'critical']),
            'warning_count': len([a for a in unique_alerts if a['level'] == 'warning']),
            'alerts': unique_alerts[:50]  # 返回前50条告警
        }

    def get_data_quality_info(self, province=None):
        """获取数据质量信息"""
        df_filtered = self.df if province is None else self.df[self.df['Province'] == province]

        # 计算缺失值
        missing_data = []
        important_columns = ['Water_Quality_Index', 'COD_mg_L', 'pH', 'Total_Nitrogen_mg_L',
                            'Total_Phosphorus_mg_L', 'Dissolved_Oxygen_mg_L']

        for col in important_columns:
            if col in df_filtered.columns:
                missing_count = df_filtered[col].isna().sum()
                missing_percent = (missing_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
                missing_data.append({
                    'indicator': col,
                    'missing_count': int(missing_count),
                    'missing_percent': round(missing_percent, 2),
                    'completeness': round(100 - missing_percent, 2)
                })

        # 计算更新频率
        if 'Date' in df_filtered.columns and len(df_filtered) > 1:
            date_diffs = df_filtered.sort_values('Date')['Date'].diff().dt.days.dropna()
            avg_interval = float(date_diffs.mean()) if len(date_diffs) > 0 else 0
            latest_update = df_filtered['Date'].max()
            days_since_update = (datetime.now() - latest_update).days if pd.notna(latest_update) else 999
        else:
            avg_interval = 0
            days_since_update = 999

        # 数据质量评分
        completeness_score = sum([item['completeness'] for item in missing_data]) / len(missing_data) if missing_data else 100
        timeliness_score = max(0, 100 - days_since_update * 5)
        quality_score = (completeness_score * 0.6 + timeliness_score * 0.4)

        return {
            'quality_score': round(quality_score, 1),
            'completeness': round(completeness_score, 1),
            'timeliness': round(timeliness_score, 1),
            'total_records': int(len(df_filtered)),
            'total_stations': int(df_filtered['Monitoring_Station'].nunique()),
            'date_range': {
                'start': df_filtered['Date'].min().strftime('%Y-%m-%d') if pd.notna(df_filtered['Date'].min()) else None,
                'end': df_filtered['Date'].max().strftime('%Y-%m-%d') if pd.notna(df_filtered['Date'].max()) else None
            },
            'latest_update': df_filtered['Date'].max().strftime('%Y-%m-%d') if pd.notna(df_filtered['Date'].max()) else None,
            'days_since_update': int(days_since_update),
            'avg_sampling_interval': round(avg_interval, 1),
            'missing_data': missing_data
        }

    def get_pollution_distribution(self, province=None, start_date=None, end_date=None):
        """获取污染等级分布数据"""
        df_filtered = self.df.copy()

        if province:
            df_filtered = df_filtered[df_filtered['Province'] == province]
        if start_date:
            df_filtered = df_filtered[df_filtered['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_filtered = df_filtered[df_filtered['Date'] <= pd.to_datetime(end_date)]

        distribution = df_filtered['Pollution_Level'].value_counts().to_dict()

        # 确保所有等级都有数据
        all_levels = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
        result = []
        for level in all_levels:
            result.append({
                'level': level,
                'count': int(distribution.get(level, 0)),
                'percentage': round((distribution.get(level, 0) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0, 1)
            })

        return {
            'distribution': result,
            'total_records': int(len(df_filtered))
        }

    def get_province_comparison(self, indicator='Water_Quality_Index'):
        """获取省份对比数据"""
        if indicator not in self.df.columns:
            return {'provinces': []}

        comparison = self.df.groupby('Province').agg({
            indicator: ['mean', 'min', 'max', 'std'],
            'Monitoring_Station': 'nunique',
            'Pollution_Level': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
        }).reset_index()

        comparison.columns = ['province', 'avg', 'min', 'max', 'std', 'station_count', 'dominant_level']

        provinces = []
        for _, row in comparison.iterrows():
            provinces.append({
                'province': row['province'],
                'average': round(float(row['avg']), 2),
                'min': round(float(row['min']), 2),
                'max': round(float(row['max']), 2),
                'std': round(float(row['std']), 2),
                'station_count': int(row['station_count']),
                'dominant_level': row['dominant_level']
            })

        # 按平均值排序
        provinces.sort(key=lambda x: x['average'])

        return {
            'indicator': indicator,
            'provinces': provinces
        }

    def get_indicator_correlation(self, province=None):
        """获取指标相关性数据"""
        df_filtered = self.df if province is None else self.df[self.df['Province'] == province]

        # 选择数值型指标
        numeric_indicators = ['Water_Quality_Index', 'COD_mg_L', 'pH', 'Total_Nitrogen_mg_L',
                             'Total_Phosphorus_mg_L', 'Dissolved_Oxygen_mg_L', 'Ammonia_Nitrogen_mg_L',
                             'Turbidity_NTU']

        available_indicators = [ind for ind in numeric_indicators if ind in df_filtered.columns]

        if len(available_indicators) < 2:
            return {'indicators': [], 'correlation_matrix': []}

        # 计算相关性矩阵
        corr_matrix = df_filtered[available_indicators].corr()

        # 转换为前端可用的格式
        correlation_data = []
        for i, ind1 in enumerate(available_indicators):
            row = []
            for j, ind2 in enumerate(available_indicators):
                row.append({
                    'x': ind1,
                    'y': ind2,
                    'value': round(float(corr_matrix.iloc[i, j]), 3)
                })
            correlation_data.append(row)

        return {
            'indicators': available_indicators,
            'correlation_matrix': correlation_data
        }

    def get_multi_indicator_trend(self, indicators, province=None, interval='month'):
        """获取多指标趋势数据"""
        df_filtered = self.df.copy()

        if province:
            df_filtered = df_filtered[df_filtered['Province'] == province]

        # 按时间间隔聚合
        if interval == 'day':
            df_filtered['period'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')
        elif interval == 'week':
            df_filtered['period'] = df_filtered['Date'].dt.strftime('%Y-W%U')
        else:  # month
            df_filtered['period'] = df_filtered['Date'].dt.strftime('%Y-%m')

        # 为每个指标计算趋势
        trends = {}
        for indicator in indicators:
            if indicator in df_filtered.columns:
                trend = df_filtered.groupby('period')[indicator].mean().reset_index()
                trend.columns = ['period', 'value']
                trends[indicator] = trend.to_dict('records')

        return {
            'indicators': indicators,
            'trends': trends
        }

    def _get_unit(self, indicator_name):
        """获取指标单位"""
        units = {
            'Water_Quality_Index': '',
            'COD_mg_L': 'mg/L',
            'pH': '',
            'Total_Nitrogen_mg_L': 'mg/L',
            'Total_Phosphorus_mg_L': 'mg/L',
            'Dissolved_Oxygen_mg_L': 'mg/L',
            'Ammonia_Nitrogen_mg_L': 'mg/L',
            'Turbidity_NTU': 'NTU',
            'BOD_mg_L': 'mg/L'
        }
        return units.get(indicator_name, '')
