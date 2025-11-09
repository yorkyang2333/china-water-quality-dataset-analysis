"""
水质指标有效范围配置
从 EDA/traditional_eda/valid_ranges_analysis.txt 提取
"""

# 基于预处理数据集的有效范围
VALID_RANGES = {
    'Water_Temperature_C': {'min': 5.4, 'max': 34.915, 'unit': '°C'},
    'pH': {'min': 5.56, 'max': 8.48, 'unit': ''},
    'Dissolved_Oxygen_mg_L': {'min': 2.15, 'max': 14.02, 'unit': 'mg/L'},
    'Conductivity_uS_cm': {'min': 153.723, 'max': 842.89, 'unit': 'μS/cm'},
    'Turbidity_NTU': {'min': 0.0, 'max': 26.57, 'unit': 'NTU'},
    'Nitrate_mg_L': {'min': 0.33, 'max': 3.7, 'unit': 'mg/L'},
    'Nitrite_mg_L': {'min': 0.0, 'max': 0.051, 'unit': 'mg/L'},
    'Ammonia_N_mg_L': {'min': 0.0, 'max': 1.253, 'unit': 'mg/L'},
    'Total_Phosphorus_mg_L': {'min': 0.0, 'max': 0.242, 'unit': 'mg/L'},
    'Total_Nitrogen_mg_L': {'min': 1.32, 'max': 4.7, 'unit': 'mg/L'},
    'COD_mg_L': {'min': 2.09, 'max': 51.093, 'unit': 'mg/L'},
    'BOD_mg_L': {'min': 0.68, 'max': 8.79, 'unit': 'mg/L'},
    'Heavy_Metals_Pb_ug_L': {'min': 0.0, 'max': 10.89, 'unit': 'μg/L'},
    'Heavy_Metals_Cd_ug_L': {'min': 0.0, 'max': 1.12, 'unit': 'μg/L'},
    'Heavy_Metals_Hg_ug_L': {'min': 0.0, 'max': 0.24, 'unit': 'μg/L'},
    'Coliform_Count_CFU_100mL': {'min': 69.0, 'max': 160.828, 'unit': 'CFU/100mL'},
}


def check_value_in_range(indicator_name: str, value: float) -> dict:
    """
    检查指标值是否在有效范围内

    Args:
        indicator_name: 指标名称
        value: 指标值

    Returns:
        {'in_range': bool, 'warning': str or None, 'range_info': str}
    """
    if indicator_name not in VALID_RANGES:
        return {
            'in_range': None,  # 无法判断
            'warning': None,
            'range_info': f'未知指标: {indicator_name}'
        }

    range_info = VALID_RANGES[indicator_name]
    min_val = range_info['min']
    max_val = range_info['max']
    unit = range_info['unit']

    if value < min_val:
        return {
            'in_range': False,
            'warning': f'⚠️ 警告: {indicator_name} 值过低\n期望范围: {min_val} - {max_val} {unit}\n实际值: {value}',
            'range_info': f'{min_val} ~ {max_val} {unit}'
        }
    elif value > max_val:
        return {
            'in_range': False,
            'warning': f'⚠️ 警告: {indicator_name} 值过高\n期望范围: {min_val} - {max_val} {unit}\n实际值: {value}',
            'range_info': f'{min_val} ~ {max_val} {unit}'
        }
    else:
        return {
            'in_range': True,
            'warning': None,
            'range_info': f'{min_val} ~ {max_val} {unit}'
        }


def check_all_values(data: dict) -> list:
    """
    检查所有输入值

    Args:
        data: 包含指标的字典

    Returns:
        警告列表
    """
    warnings = []

    for indicator_name, value in data.items():
        if indicator_name in VALID_RANGES and value is not None:
            try:
                value = float(value)
                result = check_value_in_range(indicator_name, value)
                if not result['in_range'] and result['warning']:
                    warnings.append({
                        'indicator': indicator_name,
                        'message': result['warning']
                    })
            except (ValueError, TypeError):
                pass

    return warnings
