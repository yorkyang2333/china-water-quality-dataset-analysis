"""
假设场景分析模块 (Scenario Analysis Module)
用于模拟"如果某个污染物浓度变化，其他指标会如何变化"

日期: 2025-11-14
版本: v1.0
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, List, Tuple

class ScenarioAnalyzer:
    def __init__(self):
        """初始化场景分析器"""
        self.model = None

        # 所有水质指标（14个基础指标）
        self.all_indicators = [
            'Water_Temperature_C',
            'pH',
            'Dissolved_Oxygen_mg_L',
            'Conductivity_uS_cm',
            'Turbidity_NTU',
            'Nitrate_mg_L',
            'Nitrite_mg_L',
            'Ammonia_N_mg_L',
            'Total_Phosphorus_mg_L',
            'Total_Nitrogen_mg_L',
            'COD_mg_L',
            'BOD_mg_L',
            'Coliform_Count_CFU_100mL',
            'Total_Heavy_Metals_ug_L'  # 已合并的重金属
        ]

        # 无泄露高级特征（4个）
        self.clean_features = [
            'DO_saturation_mg_L',      # 仅依赖水温
            'Alkalinity_proxy',        # 仅依赖电导率
            'Ionic_strength_proxy',    # 仅依赖电导率
            'Organic_acid_proxy'       # COD - BOD
        ]

        # 指标的历史统计范围（用于验证输入合理性）
        # 这些值应该从数据集中计算得出
        self.indicator_ranges = {
            'Water_Temperature_C': (0.0, 35.0),
            'pH': (6.0, 9.5),
            'Dissolved_Oxygen_mg_L': (0.0, 15.0),
            'Conductivity_uS_cm': (50.0, 2000.0),
            'Turbidity_NTU': (0.0, 100.0),
            'Nitrate_mg_L': (0.0, 10.0),
            'Nitrite_mg_L': (0.0, 1.0),
            'Ammonia_N_mg_L': (0.0, 5.0),
            'Total_Phosphorus_mg_L': (0.0, 1.0),
            'Total_Nitrogen_mg_L': (0.0, 15.0),
            'COD_mg_L': (0.0, 50.0),
            'BOD_mg_L': (0.0, 20.0),
            'Coliform_Count_CFU_100mL': (0.0, 100000.0),
            'Total_Heavy_Metals_ug_L': (0.0, 100.0)
        }

        # 省份历史均值（作为基线）
        self.province_baselines = self._load_province_baselines()

        # 尝试加载模型
        self._load_model()

    def _load_model(self):
        """加载无泄露版XGBoost模型"""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型路径候选
        model_paths = [
            os.path.join(current_dir, 'saved_models', 'xgboost_clean.pkl'),  # 无泄露版
            os.path.join(current_dir, 'saved_models', 'xgboost_model.pkl'),  # 分类模型（备选）
        ]

        for path in model_paths:
            full_path = os.path.normpath(path)
            if os.path.exists(full_path):
                try:
                    self.model = joblib.load(full_path)
                    print(f"[OK] Scenario analysis model loaded from: {full_path}")
                    return
                except Exception as e:
                    print(f"[WARN] Error loading model from {full_path}: {e}")

        print("[WARN] No trained clean model found. Using simplified simulation.")

    def _load_province_baselines(self) -> Dict[str, Dict[str, float]]:
        """加载各省份的历史均值作为基线"""
        # TODO: 从数据库或预计算文件中加载
        # 暂时使用全国平均值
        baseline = {
            'Water_Temperature_C': 18.5,
            'pH': 7.5,
            'Dissolved_Oxygen_mg_L': 6.5,
            'Conductivity_uS_cm': 450.0,
            'Turbidity_NTU': 8.0,
            'Nitrate_mg_L': 2.0,
            'Nitrite_mg_L': 0.05,
            'Ammonia_N_mg_L': 0.5,
            'Total_Phosphorus_mg_L': 0.1,
            'Total_Nitrogen_mg_L': 2.5,
            'COD_mg_L': 15.0,
            'BOD_mg_L': 3.0,
            'Coliform_Count_CFU_100mL': 500.0,
            'Total_Heavy_Metals_ug_L': 5.0
        }

        # 所有省份暂时使用相同基线（后续可以差异化）
        provinces = ['Beijing', 'Guangdong', 'Henan', 'Hubei', 'Jiangsu',
                    'Shandong', 'Shanghai', 'Sichuan', 'Yunnan', 'Zhejiang']

        return {province: baseline.copy() for province in provinces}

    def _generate_features(self, indicator_values: Dict[str, float]) -> np.ndarray:
        """
        生成无泄露特征向量

        参数：
            indicator_values: 14个基础指标的字典

        返回：
            特征数组（基础指标 + 时间特征 + 空间特征 + 4个无泄露高级特征）
        """
        features = []

        # 1. 基础指标（14个）
        for indicator in self.all_indicators:
            features.append(indicator_values.get(indicator, 0.0))

        # 2. 时间特征（7个，暂时使用默认值）
        features.extend([6, -0.5, 0.866, 15, 0.5, -0.866, 2])  # 6月，sin/cos编码等

        # 3. 空间特征（1个省份编码 + 2个经纬度）
        features.extend([0, 39.9, 116.4])  # 默认北京

        # 4. 无泄露高级特征（4个）
        # DO饱和度（Weiss公式）
        T = indicator_values.get('Water_Temperature_C', 20.0)
        DO_saturation = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3
        features.append(DO_saturation)

        # 碱度代理
        conductivity = indicator_values.get('Conductivity_uS_cm', 450.0)
        alkalinity = conductivity * 0.6
        features.append(alkalinity)

        # 离子强度代理
        ionic_strength = conductivity / 1000
        features.append(ionic_strength)

        # 有机酸代理
        COD = indicator_values.get('COD_mg_L', 15.0)
        BOD = indicator_values.get('BOD_mg_L', 3.0)
        organic_acid = max(0, COD - BOD)
        features.append(organic_acid)

        return np.array(features).reshape(1, -1)

    def analyze_scenario(self,
                        province: str,
                        baseline_values: Dict[str, float],
                        changes: Dict[str, float]) -> Dict:
        """
        分析假设场景：给定基线值和调整后的值，预测其他指标的变化

        参数：
            province: 省份名称
            baseline_values: 基线值（14个指标）
            changes: 用户调整的指标及变化量（百分比，如 {"COD_mg_L": 50} 表示增加50%）

        返回：
            {
                "baseline_predictions": {...},      # 基线场景的预测
                "adjusted_predictions": {...},      # 调整后场景的预测
                "changes": {...},                   # 各指标的变化量和百分比
                "sensitivity_ranking": [...],       # 敏感度排序（影响最大的指标）
                "warnings": [...]                   # 警告信息
            }
        """
        warnings = []

        # 1. 验证输入
        for indicator, value in baseline_values.items():
            if indicator in self.indicator_ranges:
                min_val, max_val = self.indicator_ranges[indicator]
                if value < min_val * 0.5 or value > max_val * 2:
                    warnings.append(f"{indicator}={value:.2f} 超出合理范围 [{min_val}, {max_val}]")

        # 2. 应用调整
        adjusted_values = baseline_values.copy()
        for indicator, change_pct in changes.items():
            if indicator in adjusted_values:
                original = adjusted_values[indicator]
                adjusted_values[indicator] = original * (1 + change_pct / 100)

        # 3. 生成特征
        baseline_features = self._generate_features(baseline_values)
        adjusted_features = self._generate_features(adjusted_values)

        # 4. 预测（如果有模型）或模拟
        if self.model is not None:
            try:
                # 使用真实模型预测
                baseline_pred = self._predict_with_model(baseline_features, baseline_values)
                adjusted_pred = self._predict_with_model(adjusted_features, adjusted_values)
            except Exception as e:
                print(f"[ERROR] Model prediction failed: {e}")
                baseline_pred = self._simulate_predictions(baseline_values)
                adjusted_pred = self._simulate_predictions(adjusted_values)
        else:
            # 使用简化模拟
            baseline_pred = self._simulate_predictions(baseline_values)
            adjusted_pred = self._simulate_predictions(adjusted_values)

        # 5. 计算变化
        changes_dict = {}
        for indicator in self.all_indicators:
            if indicator not in changes:  # 只计算未被用户调整的指标
                baseline_val = baseline_pred.get(indicator, baseline_values.get(indicator, 0))
                adjusted_val = adjusted_pred.get(indicator, adjusted_values.get(indicator, 0))

                change_abs = adjusted_val - baseline_val
                change_pct = (change_abs / baseline_val * 100) if baseline_val != 0 else 0

                changes_dict[indicator] = {
                    "baseline": baseline_val,
                    "adjusted": adjusted_val,
                    "change_abs": change_abs,
                    "change_pct": change_pct
                }

        # 6. 敏感度排序（按变化百分比的绝对值）
        sensitivity_ranking = sorted(
            changes_dict.items(),
            key=lambda x: abs(x[1]['change_pct']),
            reverse=True
        )[:5]  # 前5名

        return {
            "baseline_predictions": baseline_pred,
            "adjusted_predictions": adjusted_pred,
            "changes": changes_dict,
            "sensitivity_ranking": [
                {
                    "indicator": ind,
                    "baseline": data["baseline"],
                    "adjusted": data["adjusted"],
                    "change_abs": data["change_abs"],
                    "change_pct": data["change_pct"]
                }
                for ind, data in sensitivity_ranking
            ],
            "warnings": warnings
        }

    def _predict_with_model(self, features: np.ndarray,
                           current_values: Dict[str, float]) -> Dict[str, float]:
        """使用真实模型预测（多目标回归）"""
        # TODO: 实现真实的多目标预测
        # 暂时返回当前值
        return current_values.copy()

    def _simulate_predictions(self, indicator_values: Dict[str, float]) -> Dict[str, float]:
        """
        简化模拟预测（基于物理化学关系）
        用于在没有训练模型时提供演示
        """
        predictions = indicator_values.copy()

        # 简化的物理化学关系（示例）
        COD = indicator_values.get('COD_mg_L', 15.0)
        BOD = indicator_values.get('BOD_mg_L', 3.0)
        T = indicator_values.get('Water_Temperature_C', 20.0)

        # 1. DO与COD/BOD负相关（有机物消耗氧气）
        DO_baseline = 6.5
        DO_effect = -0.05 * (COD + BOD * 2)  # 简化关系
        predictions['Dissolved_Oxygen_mg_L'] = max(0, DO_baseline + DO_effect + (25 - T) * 0.1)

        # 2. 氨氮与总氮正相关
        TN = indicator_values.get('Total_Nitrogen_mg_L', 2.5)
        predictions['Ammonia_N_mg_L'] = TN * 0.2  # 氨氮约占总氮20%

        # 3. pH受温度和有机物影响
        pH_baseline = 7.5
        pH_effect = -0.01 * COD + 0.005 * (T - 20)
        predictions['pH'] = max(6.0, min(9.0, pH_baseline + pH_effect))

        # 4. 浊度与胶体和悬浮物相关（简化为与COD相关）
        predictions['Turbidity_NTU'] = max(0, COD * 0.3 + 2)

        # 5. 电导率与总溶解性固体相关（简化）
        predictions['Conductivity_uS_cm'] = 300 + TN * 20 + predictions['Total_Phosphorus_mg_L'] * 100

        return predictions

    def get_baseline_for_province(self, province: str) -> Dict[str, float]:
        """获取指定省份的基线值（历史均值）"""
        return self.province_baselines.get(province, self.province_baselines['Beijing'])

    def get_preset_scenarios(self) -> List[Dict]:
        """获取预设场景列表"""
        return [
            {
                "name": "减排效果模拟",
                "description": "模拟工业减排：COD和BOD同时降低30%",
                "changes": {
                    "COD_mg_L": -30,
                    "BOD_mg_L": -30
                }
            },
            {
                "name": "极端污染事故",
                "description": "模拟突发污染：重金属浓度增加10倍",
                "changes": {
                    "Total_Heavy_Metals_ug_L": 1000
                }
            },
            {
                "name": "夏季高温影响",
                "description": "模拟夏季高温：水温升高10°C",
                "changes": {
                    "Water_Temperature_C": 10
                }
            },
            {
                "name": "富营养化风险",
                "description": "模拟农业径流：总氮和总磷增加50%",
                "changes": {
                    "Total_Nitrogen_mg_L": 50,
                    "Total_Phosphorus_mg_L": 50
                }
            },
            {
                "name": "冬季低温影响",
                "description": "模拟冬季低温：水温降低10°C",
                "changes": {
                    "Water_Temperature_C": -10
                }
            }
        ]


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  假设场景分析模块测试")
    print("="*70)

    analyzer = ScenarioAnalyzer()

    # 测试1：获取北京基线
    print("\n[TEST 1] 获取北京省基线值:")
    baseline = analyzer.get_baseline_for_province("Beijing")
    print(f"COD: {baseline['COD_mg_L']:.2f} mg/L")
    print(f"DO: {baseline['Dissolved_Oxygen_mg_L']:.2f} mg/L")
    print(f"pH: {baseline['pH']:.2f}")

    # 测试2：减排场景
    print("\n[TEST 2] 减排效果模拟（COD/BOD -30%）:")
    result = analyzer.analyze_scenario(
        province="Beijing",
        baseline_values=baseline,
        changes={"COD_mg_L": -30, "BOD_mg_L": -30}
    )

    print("\n敏感度排名（前5）:")
    for rank, item in enumerate(result['sensitivity_ranking'], 1):
        print(f"  {rank}. {item['indicator']}: "
              f"{item['baseline']:.2f} -> {item['adjusted']:.2f} "
              f"({item['change_pct']:+.1f}%)")

    # 测试3：预设场景
    print("\n[TEST 3] 预设场景列表:")
    scenarios = analyzer.get_preset_scenarios()
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario['name']}: {scenario['description']}")

    print("\n[DONE] 测试完成")
