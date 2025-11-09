"""
季节性趋势分析模块
用于水质指标的季节性特征分析和可视化
"""

import pandas as pd
import numpy as np
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    pass

warnings.filterwarnings('ignore')

# 水质指标列表
WATER_QUALITY_INDICATORS = [
    'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
    'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
    'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
    'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
    'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
]


class SeasonalityAnalyzer:
    """季节性分析器"""

    def __init__(self, data: pd.DataFrame):
        """
        初始化分析器

        Args:
            data: 包含 Date 列和水质指标的 DataFrame
        """
        self.data = data.copy()
        self._prepare_data()

    def _prepare_data(self):
        """准备数据"""
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])

    def prepare_timeseries(self, indicator: str, group_by: str = None, resample_freq: str = 'M'):
        """
        准备时间序列数据

        Args:
            indicator: 水质指标名称
            group_by: 分组方式 ('Province' 或 'Monitoring_Station')
            resample_freq: 重采样频率 ('D'=日, 'W'=周, 'M'=月)

        Returns:
            时间序列字典 {group_name: series}
        """
        if indicator not in self.data.columns:
            return {}

        timeseries_dict = {}

        if group_by and group_by in self.data.columns:
            groups = self.data[group_by].unique()
            for group in groups:
                group_data = self.data[self.data[group_by] == group].copy()

                # 检查数据量
                if len(group_data) < 30 or group_data[indicator].isna().sum() > len(group_data) * 0.5:
                    continue

                # 设置日期索引并重采样
                group_data = group_data.set_index('Date')
                ts = group_data[indicator].resample(resample_freq).mean()

                if len(ts) >= 24:  # 至少需要2年的月度数据
                    timeseries_dict[str(group)] = ts.dropna()
        else:
            # 全部数据
            data = self.data.set_index('Date')
            ts = data[indicator].resample(resample_freq).mean()
            if len(ts) >= 24:
                timeseries_dict['All'] = ts.dropna()

        return timeseries_dict

    def analyze_seasonality(self, timeseries: pd.Series, period: int = 12) -> Dict:
        """
        分析季节性

        Args:
            timeseries: 时间序列数据
            period: 周期（月度数据通常为12）

        Returns:
            包含季节性分析结果的字典
        """
        try:
            if len(timeseries) < 2 * period:
                return {
                    'has_seasonality': False,
                    'message': '数据不足以进行季节性分析',
                    'data_points': len(timeseries),
                    'required_points': 2 * period
                }

            # 季节性分解（移除 extrapolate 参数以兼容不同版本的 statsmodels）
            try:
                # 首先尝试使用 extrapolate 参数（新版本）
                decomposition = seasonal_decompose(timeseries, model='additive', period=period, extrapolate='fill_zeros')
            except TypeError:
                # 如果不支持 extrapolate 参数，使用基础参数
                decomposition = seasonal_decompose(timeseries, model='additive', period=period)

            # 计算季节性强度
            seasonal_strength = self._calculate_seasonality_strength(decomposition)
            trend_strength = self._calculate_trend_strength(decomposition)

            # 季节性特征
            seasonal_component = decomposition.seasonal
            seasonal_amplitude = seasonal_component.max() - seasonal_component.min()
            seasonal_std = seasonal_component.std()

            return {
                'has_seasonality': seasonal_strength > 0.3,
                'seasonality_strength': round(seasonal_strength, 3),
                'trend_strength': round(trend_strength, 3),
                'seasonal_amplitude': round(seasonal_amplitude, 3),
                'seasonal_std': round(seasonal_std, 3),
                'data_points': len(timeseries),
                'period': period,
                'trend': decomposition.trend.tail(12).tolist() if hasattr(decomposition.trend, 'tail') else []
            }
        except Exception as e:
            return {
                'has_seasonality': False,
                'error': str(e),
                'message': f'分析失败: {str(e)}'
            }

    def _calculate_seasonality_strength(self, decomposition) -> float:
        """计算季节性强度"""
        try:
            residual_var = decomposition.resid.var()
            seasonal_var = decomposition.seasonal.var()

            if residual_var + seasonal_var == 0:
                return 0.0

            strength = max(0, 1 - residual_var / (seasonal_var + residual_var))
            return min(1.0, strength)
        except:
            return 0.0

    def _calculate_trend_strength(self, decomposition) -> float:
        """计算趋势强度"""
        try:
            residual_var = decomposition.resid.var()
            trend_var = decomposition.trend.var()

            if residual_var + trend_var == 0:
                return 0.0

            strength = max(0, 1 - residual_var / (trend_var + residual_var))
            return min(1.0, strength)
        except:
            return 0.0

    def analyze_indicator(self, indicator: str, group_by: str = None) -> Dict:
        """
        分析单个指标的季节性

        Args:
            indicator: 水质指标
            group_by: 分组方式

        Returns:
            分析结果字典
        """
        if indicator not in WATER_QUALITY_INDICATORS:
            return {'error': f'未知指标: {indicator}'}

        timeseries_dict = self.prepare_timeseries(indicator, group_by)

        if not timeseries_dict:
            return {
                'indicator': indicator,
                'error': '数据不足',
                'message': '无法进行季节性分析'
            }

        results = {
            'indicator': indicator,
            'group_by': group_by,
            'groups': {}
        }

        for group_name, ts in timeseries_dict.items():
            analysis_result = self.analyze_seasonality(ts)
            results['groups'][group_name] = analysis_result

        # 计算汇总统计
        seasonality_scores = [
            v.get('seasonality_strength', 0)
            for v in results['groups'].values()
            if 'seasonality_strength' in v
        ]

        if seasonality_scores:
            results['summary'] = {
                'avg_seasonality_strength': round(np.mean(seasonality_scores), 3),
                'max_seasonality_strength': round(np.max(seasonality_scores), 3),
                'min_seasonality_strength': round(np.min(seasonality_scores), 3),
                'groups_count': len(results['groups'])
            }

        return results


def analyze_water_quality_seasonality(data: pd.DataFrame, indicator: str = None,
                                     group_by: str = 'Province') -> Dict:
    """
    分析水质指标的季节性

    Args:
        data: 水质数据 DataFrame
        indicator: 指标名称 (如果为None，分析所有指标)
        group_by: 分组方式 ('Province' 或 'Monitoring_Station')

    Returns:
        分析结果字典
    """
    analyzer = SeasonalityAnalyzer(data)

    if indicator:
        return analyzer.analyze_indicator(indicator, group_by)
    else:
        # 分析所有指标（快速模式）
        results = {
            'timestamp': datetime.now().isoformat(),
            'group_by': group_by,
            'indicators': {}
        }

        for ind in WATER_QUALITY_INDICATORS[:8]:  # 分析前8个指标加快速度
            if ind in data.columns:
                results['indicators'][ind] = analyzer.analyze_indicator(ind, group_by)

        return results
