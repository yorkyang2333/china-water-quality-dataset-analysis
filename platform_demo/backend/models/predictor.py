"""
水质预测模块
加载训练好的模型并提供预测服务
"""

import numpy as np
import pandas as pd
import os
import pickle
import joblib

class WaterQualityPredictor:
    def __init__(self):
        """初始化预测器，加载训练好的模型"""
        self.model = None
        # 模型训练时使用的特征列（按顺序）
        self.feature_columns = [
            'Latitude',
            'Longitude',
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
            'Total_Heavy_Metals_ug_L',
            'COD_BOD_Ratio',
            'NP_Ratio'
        ]
        self.pollution_levels = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']

        # 尝试加载已训练的模型
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        # 模型文件应该放在 backend/models/saved_models/ 目录下
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 优先从本地saved_models目录加载
        model_paths = [
            # 首选：backend/models/saved_models/
            os.path.join(current_dir, 'saved_models', 'xgboost_model.pkl'),
            # 备选：backend/models/
            os.path.join(current_dir, 'xgboost_model.pkl'),
            # 后备：项目根目录的Outputs（仅用于开发）
            os.path.join(current_dir, '../../../Outputs/Classification_XGBoost/xgboost_model.pkl'),
        ]

        for path in model_paths:
            full_path = os.path.normpath(path)
            if os.path.exists(full_path):
                try:
                    self.model = joblib.load(full_path)
                    print(f"✓ Model loaded successfully from: {full_path}")
                    return
                except Exception as e:
                    print(f"⚠ Error loading model from {full_path}: {e}")

        print("⚠ Warning: No trained model found. Using mock predictions.")
        print("  To use a trained model:")
        print("  1. Train the model: python models/pollution_level_classification/classification_xgboost.py")
        print("  2. Copy the model file to: platform_demo/backend/models/saved_models/xgboost_model.pkl")
        print("     Or copy to: platform_demo/backend/models/xgboost_model.pkl")

    def predict_pollution_level(self, input_data):
        """
        预测污染等级
        输入：水质指标字典
        输出：预测的污染等级及概率
        """
        try:
            # 字段名映射（支持驼峰命名和下划线命名）
            field_mapping = {
                'waterQualityIndex': 'Water_Quality_Index',
                'cod': 'COD_mg_L',
                'bod': 'BOD_mg_L',
                'totalNitrogen': 'Total_Nitrogen_mg_L',
                'totalPhosphorus': 'Total_Phosphorus_mg_L',
                'dissolvedOxygen': 'Dissolved_Oxygen_mg_L',
                'ph': 'pH',
                'ammoniaN': 'Ammonia_N_mg_L',
                'turbidity': 'Turbidity_NTU',
                'temperature': 'Water_Temperature_C',
                'waterTemperature': 'Water_Temperature_C',
                'conductivity': 'Conductivity_uS_cm',
                'nitrate': 'Nitrate_mg_L',
                'nitrite': 'Nitrite_mg_L',
                'coliformCount': 'Coliform_Count_CFU_100mL',
                'latitude': 'Latitude',
                'longitude': 'Longitude',
                'totalHeavyMetals': 'Total_Heavy_Metals_ug_L',
                'codBodRatio': 'COD_BOD_Ratio',
                'npRatio': 'NP_Ratio'
            }

            # 转换输入数据
            normalized_data = {}
            for key, value in input_data.items():
                # 如果是驼峰命名，转换为下划线命名
                if key in field_mapping:
                    normalized_data[field_mapping[key]] = value
                else:
                    normalized_data[key] = value

            # 设置默认值（基于数据集的典型值）
            default_values = {
                'Latitude': 30.0,  # 中国中部纬度
                'Longitude': 110.0,  # 中国中部经度
                'Water_Temperature_C': 18.0,
                'pH': normalized_data.get('pH', 7.0),
                'Dissolved_Oxygen_mg_L': normalized_data.get('Dissolved_Oxygen_mg_L', 7.0),
                'Conductivity_uS_cm': 500.0,
                'Turbidity_NTU': normalized_data.get('Turbidity_NTU', 5.0),
                'Nitrate_mg_L': 2.0,
                'Nitrite_mg_L': 0.1,
                'Ammonia_N_mg_L': 0.5,
                'Total_Phosphorus_mg_L': normalized_data.get('Total_Phosphorus_mg_L', 0.1),
                'Total_Nitrogen_mg_L': normalized_data.get('Total_Nitrogen_mg_L', 2.0),
                'COD_mg_L': normalized_data.get('COD_mg_L', 15.0),
                'BOD_mg_L': 3.0,
                'Coliform_Count_CFU_100mL': 100.0,
                'Total_Heavy_Metals_ug_L': 5.0,
                'COD_BOD_Ratio': 5.0,
                'NP_Ratio': 20.0
            }

            # 计算派生特征
            cod_val = normalized_data.get('COD_mg_L', default_values['COD_mg_L'])
            bod_val = normalized_data.get('BOD_mg_L', default_values['BOD_mg_L'])
            tn_val = normalized_data.get('Total_Nitrogen_mg_L', default_values['Total_Nitrogen_mg_L'])
            tp_val = normalized_data.get('Total_Phosphorus_mg_L', default_values['Total_Phosphorus_mg_L'])

            if 'COD_BOD_Ratio' not in normalized_data:
                normalized_data['COD_BOD_Ratio'] = cod_val / bod_val if bod_val > 0 else default_values['COD_BOD_Ratio']

            if 'NP_Ratio' not in normalized_data:
                normalized_data['NP_Ratio'] = tn_val / tp_val if tp_val > 0 else default_values['NP_Ratio']

            # 准备特征数据，按正确顺序
            features = []
            for col in self.feature_columns:
                if col in normalized_data:
                    value = normalized_data[col]
                else:
                    value = default_values.get(col, 0)
                features.append(float(value))

            features_array = np.array(features).reshape(1, -1)

            if self.model is not None:
                # 使用训练好的模型预测
                prediction_num = self.model.predict(features_array)[0]
                probabilities = self.model.predict_proba(features_array)[0]

                # 标签映射 (模型返回的是数字0-4)
                label_mapping = {
                    0: 'Excellent',
                    1: 'Fair',
                    2: 'Good',
                    3: 'Poor',
                    4: 'Very Poor'
                }

                prediction = label_mapping.get(int(prediction_num), 'Unknown')

                # 获取概率分布（将数字键转换为文字标签）
                prob_dict = {}
                for idx, prob in enumerate(probabilities):
                    label = label_mapping.get(idx, f'Level_{idx}')
                    prob_dict[label] = float(prob)

            else:
                # Mock预测（基于主要指标的简单规则）
                cod = normalized_data.get('COD_mg_L', default_values['COD_mg_L'])
                do = normalized_data.get('Dissolved_Oxygen_mg_L', default_values['Dissolved_Oxygen_mg_L'])
                tn = normalized_data.get('Total_Nitrogen_mg_L', default_values['Total_Nitrogen_mg_L'])

                # 基于多个指标的综合评估
                score = 0
                if cod > 30: score += 2
                elif cod > 20: score += 1

                if do < 5: score += 2
                elif do < 6: score += 1

                if tn > 5: score += 2
                elif tn > 2: score += 1

                if score >= 5:
                    prediction = 'Very Poor'
                    prob_dict = {'Excellent': 0.01, 'Good': 0.01, 'Fair': 0.03, 'Poor': 0.15, 'Very Poor': 0.80}
                elif score >= 4:
                    prediction = 'Poor'
                    prob_dict = {'Excellent': 0.01, 'Good': 0.03, 'Fair': 0.15, 'Poor': 0.70, 'Very Poor': 0.11}
                elif score >= 2:
                    prediction = 'Fair'
                    prob_dict = {'Excellent': 0.02, 'Good': 0.15, 'Fair': 0.70, 'Poor': 0.10, 'Very Poor': 0.03}
                elif score >= 1:
                    prediction = 'Good'
                    prob_dict = {'Excellent': 0.10, 'Good': 0.75, 'Fair': 0.10, 'Poor': 0.03, 'Very Poor': 0.02}
                else:
                    prediction = 'Excellent'
                    prob_dict = {'Excellent': 0.85, 'Good': 0.10, 'Fair': 0.03, 'Poor': 0.01, 'Very Poor': 0.01}

            return {
                'prediction': prediction,
                'probabilities': prob_dict,
                'confidence': float(max(prob_dict.values()))
            }

        except Exception as e:
            print(f"Prediction error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Prediction error: {str(e)}")

    def predict_water_quality(self, input_data):
        """
        预测水质指标
        输入：地点+时间信息
        输出：预测的水质指标值

        注：这是一个简化的实现，实际需要时序模型
        """
        try:
            province = input_data.get('province', 'Beijing')
            date = input_data.get('date')

            # Mock预测：基于历史平均值 + 随机波动
            # 实际应用中应使用时序模型（LSTM/Prophet等）
            base_values = {
                'Water_Quality_Index': 35.0,
                'COD_mg_L': 20.0,
                'BOD_mg_L': 4.0,
                'Total_Nitrogen_mg_L': 3.0,
                'Total_Phosphorus_mg_L': 0.1,
                'Dissolved_Oxygen_mg_L': 8.0,
                'pH': 7.0,
                'Ammonia_N_mg_L': 0.5,
                'Turbidity_NTU': 5.0
            }

            # 添加随机波动（±10%）
            np.random.seed(hash(province + str(date)) % 2**32)
            predictions = {}
            for indicator, base_value in base_values.items():
                fluctuation = np.random.uniform(-0.1, 0.1)
                predictions[indicator] = float(base_value * (1 + fluctuation))

            return {
                'province': province,
                'date': date,
                'predictions': predictions,
                'note': 'This is a mock prediction. Real implementation requires time-series model.'
            }

        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

    def get_feature_importance(self):
        """获取特征重要性（如果模型支持）"""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            importance = {}
            for col, imp in zip(self.feature_columns, self.model.feature_importances_):
                importance[col] = float(imp)
            return importance
        return None
