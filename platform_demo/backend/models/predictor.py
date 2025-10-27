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
        self.feature_columns = [
            'Water_Quality_Index', 'COD_mg_L', 'BOD_mg_L', 'Total_Nitrogen_mg_L',
            'Total_Phosphorus_mg_L', 'Dissolved_Oxygen_mg_L', 'pH',
            'Ammonia_N_mg_L', 'Turbidity_NTU', 'Temperature_C',
            'Conductivity_uS_cm', 'Lead_ug_L', 'Cadmium_ug_L', 'Mercury_ug_L',
            'Total_Heavy_Metals_ug_L', 'COD_BOD_Ratio', 'NP_Ratio'
        ]
        self.pollution_levels = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']

        # 尝试加载已训练的模型
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        # 尝试加载XGBoost模型（最佳模型）
        model_paths = [
            '../Outputs/Classification_XGBoost/xgboost_model.pkl',
            '../../Outputs/Classification_XGBoost/xgboost_model.pkl',
            '../../../Outputs/Classification_XGBoost/xgboost_model.pkl'
        ]

        for path in model_paths:
            full_path = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(full_path):
                try:
                    self.model = joblib.load(full_path)
                    print(f"Model loaded successfully from: {full_path}")
                    return
                except Exception as e:
                    print(f"Error loading model from {full_path}: {e}")

        print("Warning: No trained model found. Using mock predictions.")

    def predict_pollution_level(self, input_data):
        """
        预测污染等级
        输入：水质指标字典
        输出：预测的污染等级及概率
        """
        try:
            # 准备特征数据
            features = []
            for col in self.feature_columns:
                value = input_data.get(col, 0)
                features.append(float(value))

            features_array = np.array(features).reshape(1, -1)

            if self.model is not None:
                # 使用训练好的模型预测
                prediction = self.model.predict(features_array)[0]
                probabilities = self.model.predict_proba(features_array)[0]

                # 获取概率分布
                prob_dict = {}
                for level, prob in zip(self.model.classes_, probabilities):
                    prob_dict[level] = float(prob)

            else:
                # Mock预测（基于WQI简单规则）
                wqi = input_data.get('Water_Quality_Index', 50)
                if wqi < 20:
                    prediction = 'Excellent'
                    prob_dict = {'Excellent': 0.85, 'Good': 0.10, 'Fair': 0.03, 'Poor': 0.01, 'Very Poor': 0.01}
                elif wqi < 40:
                    prediction = 'Good'
                    prob_dict = {'Excellent': 0.10, 'Good': 0.75, 'Fair': 0.10, 'Poor': 0.03, 'Very Poor': 0.02}
                elif wqi < 60:
                    prediction = 'Fair'
                    prob_dict = {'Excellent': 0.02, 'Good': 0.15, 'Fair': 0.70, 'Poor': 0.10, 'Very Poor': 0.03}
                elif wqi < 80:
                    prediction = 'Poor'
                    prob_dict = {'Excellent': 0.01, 'Good': 0.03, 'Fair': 0.15, 'Poor': 0.70, 'Very Poor': 0.11}
                else:
                    prediction = 'Very Poor'
                    prob_dict = {'Excellent': 0.01, 'Good': 0.01, 'Fair': 0.03, 'Poor': 0.15, 'Very Poor': 0.80}

            return {
                'prediction': prediction,
                'probabilities': prob_dict,
                'confidence': float(max(prob_dict.values()))
            }

        except Exception as e:
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
