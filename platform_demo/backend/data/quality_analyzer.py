"""
数据质量分析器
分析原始水质监测数据集的质量指标
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


class DataQualityAnalyzer:
    """数据质量分析器"""

    def __init__(self, data_path: str):
        """初始化分析器"""
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self._prepare_data()

    def _prepare_data(self):
        """准备数据"""
        # 转换日期列
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

    def get_overall_metrics(self) -> Dict[str, float]:
        """
        获取总体数据质量指标

        Returns:
            包含完整度、准确度、及时性、一致性的字典
        """
        # 排除 Remarks 列计算缺失值
        df_for_completeness = self.df.drop(columns=['Remarks'], errors='ignore')
        total_cells = df_for_completeness.size
        null_cells = df_for_completeness.isnull().sum().sum()

        # 完整度：非空值比例
        completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0

        # 准确度：基于数值范围的合理性检查
        accuracy = self._calculate_accuracy()

        # 及时性：基于日期连续性
        timeliness = self._calculate_timeliness()

        # 一致性：基于数据分布的一致性
        consistency = self._calculate_consistency()

        return {
            'completeness': round(completeness, 2),
            'accuracy': round(accuracy, 2),
            'timeliness': round(timeliness, 2),
            'consistency': round(consistency, 2)
        }

    def _calculate_accuracy(self) -> float:
        """
        计算准确度（模拟preprocess.py中的完整清理过程）

        准确度 = 保留的行数 / 原始行数 * 100

        清理过程分两步：
        1. 按valid_ranges过滤
        2. 3-sigma和IQR双重异常值检测
        """
        df_temp = self.df.copy()
        original_rows = len(df_temp)

        # 第一步：按valid_ranges过滤（与preprocess.py第98-111行一致）
        valid_ranges = {
            'Water_Temperature_C': (0, 40),
            'pH': (4.5, 10.0),
            'Dissolved_Oxygen_mg_L': (0, 20),
            'Conductivity_uS_cm': (50, 2000),
            'Turbidity_NTU': (0, 150),
            'Nitrate_mg_L': (0, 100),
            'Nitrite_mg_L': (0, 5),
            'Ammonia_N_mg_L': (0, 10),
            'Total_Phosphorus_mg_L': (0, 2),
            'Total_Nitrogen_mg_L': (0, 20),
            'COD_mg_L': (0, 300),
            'BOD_mg_L': (0, 100),
            'Heavy_Metals_Pb_ug_L': (0, 100),
            'Heavy_Metals_Cd_ug_L': (0, 10),
            'Heavy_Metals_Hg_ug_L': (0, 2),
            'Coliform_Count_CFU_100mL': (0, 10000)
        }

        for col, (min_val, max_val) in valid_ranges.items():
            if col in df_temp.columns:
                # 转换为数值，错误值变为NaN
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                # 移除NaN
                df_temp.dropna(subset=[col], inplace=True)
                # 按范围过滤
                df_temp = df_temp[(df_temp[col] >= min_val) & (df_temp[col] <= max_val)]

        after_range_filter = len(df_temp)

        # 第二步：异常值检测（与preprocess.py第130-153行一致）
        numerical_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        if 'Remarks' in numerical_cols:
            numerical_cols.remove('Remarks')

        # 初始化行掩码
        rows_to_keep = pd.Series([True] * len(df_temp), index=df_temp.index)

        for col in numerical_cols:
            if df_temp[col].notna().sum() < 2:  # 需要至少2个值来计算统计量
                continue

            # 3-sigma方法
            mean = df_temp[col].mean()
            std = df_temp[col].std()
            lower_bound_3sigma = mean - 3 * std
            upper_bound_3sigma = mean + 3 * std

            # IQR方法
            Q1 = df_temp[col].quantile(0.25)
            Q3 = df_temp[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR

            # 取更宽松的范围（并集）
            final_lower_bound = min(lower_bound_3sigma, lower_bound_iqr)
            final_upper_bound = max(upper_bound_3sigma, upper_bound_iqr)

            # 更新掩码
            rows_to_keep &= (df_temp[col] >= final_lower_bound) & (df_temp[col] <= final_upper_bound)

        df_cleaned = df_temp[rows_to_keep]
        final_rows = len(df_cleaned)

        # 计算保留的行数比例作为准确度
        accuracy = (final_rows / original_rows * 100) if original_rows > 0 else 100.0
        return float(accuracy)

    def _calculate_timeliness(self) -> float:
        """计算及时性"""
        if 'Date' not in self.df.columns:
            return 100.0

        # 计算数据的时间跨度覆盖率
        df_with_date = self.df.dropna(subset=['Date'])
        if len(df_with_date) == 0:
            return 0.0

        date_range = (df_with_date['Date'].max() - df_with_date['Date'].min()).days
        expected_days = date_range
        actual_days = df_with_date['Date'].nunique()

        # 计算覆盖率（不是每天都有数据是正常的）
        coverage = (actual_days / expected_days * 100) if expected_days > 0 else 100.0

        # 归一化到合理范围 (假设20%的覆盖率是基准)
        normalized_score = min(coverage / 0.2 * 100, 100.0)

        return normalized_score

    def _calculate_consistency(self) -> float:
        """计算一致性"""
        # 检查不同站点之间数据分布的一致性
        if 'Monitoring_Station' not in self.df.columns:
            return 100.0

        consistency_scores = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:5]:  # 只检查前5个数值列以提高性能
            if col in self.df.columns and self.df[col].notna().sum() > 0:
                # 计算变异系数 (CV) - 越小越一致
                cv = self.df[col].std() / self.df[col].mean() if self.df[col].mean() != 0 else 0
                # 转换为得分 (CV越小得分越高)
                score = max(0, 100 - cv * 10)
                consistency_scores.append(score)

        return np.mean(consistency_scores) if consistency_scores else 100.0

    def get_station_quality(self, province: str = None) -> List[Dict[str, Any]]:
        """
        获取各站点的数据质量

        Args:
            province: 省份筛选（可选）

        Returns:
            站点质量列表
        """
        df_filtered = self.df.copy()
        if province:
            df_filtered = df_filtered[df_filtered['Province'] == province]

        station_quality = []

        for station in df_filtered['Monitoring_Station'].unique():
            station_data = df_filtered[df_filtered['Monitoring_Station'] == station]

            # 计算该站点的完整度（排除 Remarks 列）
            station_data_for_completeness = station_data.drop(columns=['Remarks'], errors='ignore')
            total_cells = station_data_for_completeness.size
            null_cells = station_data_for_completeness.isnull().sum().sum()
            completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0

            # 计算准确度 (简化版)
            numeric_cols = station_data.select_dtypes(include=[np.number]).columns
            valid_ratio = []
            for col in numeric_cols:
                if station_data[col].notna().sum() > 0:
                    # 使用IQR方法检测异常值
                    Q1 = station_data[col].quantile(0.25)
                    Q3 = station_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    valid = station_data[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                    valid_ratio.append(valid.sum() / valid.count() * 100)

            accuracy = np.mean(valid_ratio) if valid_ratio else 100.0

            # 缺失字段分析
            missing_fields = []
            for col in station_data.columns:
                if station_data[col].isnull().sum() > 0:
                    missing_fields.append(col)

            # 获取省份
            province_name = station_data['Province'].iloc[0] if 'Province' in station_data.columns else 'Unknown'

            # 最后更新时间
            last_update = station_data['Date'].max() if 'Date' in station_data.columns else None
            last_update_str = last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else 'N/A'

            station_quality.append({
                'station_name': station,
                'province': province_name,
                'data_count': len(station_data),
                'completeness': round(completeness, 2),
                'accuracy': round(accuracy, 2),
                'last_update': last_update_str,
                'missing_fields': ', '.join(missing_fields[:3]) if missing_fields else '无'
            })

        return station_quality

    def get_quality_issues(self, province: str = None) -> List[Dict[str, Any]]:
        """
        获取数据质量问题

        Args:
            province: 省份筛选（可选）

        Returns:
            质量问题列表
        """
        df_filtered = self.df.copy()
        if province:
            df_filtered = df_filtered[df_filtered['Province'] == province]

        issues = []

        # 检测缺失值问题
        for col in df_filtered.columns:
            null_count = df_filtered[col].isnull().sum()
            if null_count > 0:
                null_ratio = null_count / len(df_filtered) * 100
                if null_ratio > 5:  # 缺失率超过5%才报告
                    severity = '严重' if null_ratio > 20 else '中等' if null_ratio > 10 else '轻微'
                    issues.append({
                        'station': f'{null_count}个站点',
                        'issue': f'{col}字段缺失数据（{null_ratio:.1f}%）',
                        'severity': severity,
                        'detected_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                    })

        # 检测异常值问题（pH范围）
        if 'pH' in df_filtered.columns:
            abnormal_ph = df_filtered[(df_filtered['pH'] < 6) | (df_filtered['pH'] > 9)]
            if len(abnormal_ph) > 0:
                stations_affected = abnormal_ph['Monitoring_Station'].nunique()
                issues.append({
                    'station': f'{stations_affected}个站点',
                    'issue': f'pH值异常（{len(abnormal_ph)}条记录）',
                    'severity': '中等',
                    'detected_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                })

        # 检测重复记录
        duplicates = df_filtered.duplicated(subset=['Monitoring_Station', 'Date'], keep=False)
        if duplicates.sum() > 0:
            issues.append({
                'station': '多个站点',
                'issue': f'发现{duplicates.sum()}条重复记录',
                'severity': '轻微',
                'detected_at': datetime.now().strftime('%Y-%m-%d %H:%M')
            })

        return issues[:20]  # 返回前20个问题

    def get_completeness_trend(self) -> Dict[str, List]:
        """
        获取数据完整度趋势

        Returns:
            包含labels和values的字典
        """
        if 'Date' not in self.df.columns:
            return {'labels': [], 'values': []}

        df_with_date = self.df.copy()
        df_with_date['Date'] = pd.to_datetime(df_with_date['Date'], errors='coerce')
        df_with_date = df_with_date.dropna(subset=['Date'])

        # 按月统计完整度
        df_with_date['YearMonth'] = df_with_date['Date'].dt.to_period('M')

        monthly_completeness = []
        labels = []

        for period in df_with_date['YearMonth'].unique():
            month_data = df_with_date[df_with_date['YearMonth'] == period]
            # 排除 Remarks 列计算完整度
            month_data_for_completeness = month_data.drop(columns=['Remarks'], errors='ignore')
            total_cells = month_data_for_completeness.size
            null_cells = month_data_for_completeness.isnull().sum().sum()
            completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0

            monthly_completeness.append(round(completeness, 2))
            labels.append(str(period))

        # 只返回最近30个月
        return {
            'labels': labels[-30:],
            'values': monthly_completeness[-30:]
        }
