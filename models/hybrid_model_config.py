"""
LSTM + 随机森林混合模型 - 配置和工具模块
Configuration and utilities for LSTM + RF hybrid models

作者：Claude Code
日期：2025-11-09
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

class HybridModelConfig:
    """混合模型配置"""

    # LSTM超参数
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 16
    LSTM_SEQ_LENGTH = 10
    LSTM_LEARNING_RATE = 0.001

    # 随机森林超参数
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2

    # 数据处理
    DATA_SPLIT_YEAR = 2024  # 2023训练，2024验证
    MIN_TRAIN_SAMPLES = 30
    MIN_TEST_SAMPLES = 5

    # 融合权重（LSTM:RF）
    LSTM_WEIGHT = 0.5
    RF_WEIGHT = 0.5

    # 特征工程
    LAG_DAYS = [1, 3, 7, 14, 30]
    ROLLING_WINDOWS = [7, 14, 30]

    # 输出配置
    OUTPUT_BASE = 'Outputs'
    PLOT_DPI = 150
    PLOT_FIGSIZE = (14, 10)

    # 目标指标
    TARGET_INDICATORS = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L',
        'COD_mg_L', 'BOD_mg_L', 'Heavy_Metals_Pb_ug_L'
    ]

class DataLocator:
    """数据文件定位器"""

    @staticmethod
    def find_data_file():
        """查找数据文件的位置"""
        search_paths = [
            # 根目录
            '中国水污染监测数据集.csv',
            'preprocessed_data.csv',
            'preprocessed_data_feature_eng.csv',
            # 上级目录
            '../中国水污染监测数据集.csv',
            '../preprocessed_data.csv',
            '../preprocessed_data_feature_eng.csv',
            # 上上级目录
            '../../中国水污染监测数据集.csv',
            '../../preprocessed_data.csv',
            # EDA目录
            '../EDA/中国水污染监测数据集.csv',
            '../../EDA/中国水污染监测数据集.csv',
            # preprocess目录
            '../preprocess/preprocessed_data.csv',
            '../../preprocess/preprocessed_data.csv',
        ]

        for path in search_paths:
            if os.path.exists(path):
                full_path = os.path.abspath(path)
                return full_path

        return None

    @staticmethod
    def copy_data_to_models():
        """将数据复制到models文件夹的data目录"""
        src_file = DataLocator.find_data_file()
        if not src_file:
            print("❌ 找不到数据文件")
            return False

        model_data_dir = Path('data')
        model_data_dir.mkdir(parents=True, exist_ok=True)

        dst_file = model_data_dir / 'water_quality_data.csv'

        try:
            print(f"📋 从 {src_file} 复制数据...")
            df = pd.read_csv(src_file)
            df.to_csv(dst_file, index=False, encoding='utf-8')
            print(f"✓ 数据已保存到 {dst_file}")
            print(f"  - 行数: {len(df)}")
            print(f"  - 列数: {len(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ 复制数据出错: {e}")
            return False

class ModelUtils:
    """模型工具函数"""

    @staticmethod
    def validate_data(df, indicator):
        """验证数据是否有效"""
        if indicator not in df.columns:
            return False, f"指标 {indicator} 不在数据中"

        non_null = df[indicator].notna().sum()
        if non_null < 30:
            return False, f"有效数据不足 ({non_null} < 30)"

        return True, "数据有效"

    @staticmethod
    def create_output_dirs(base_name):
        """创建输出目录"""
        output_dir = f"{HybridModelConfig.OUTPUT_BASE}/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/Plots", exist_ok=True)
        os.makedirs(f"{output_dir}/Reports", exist_ok=True)
        return output_dir

    @staticmethod
    def save_report(filepath, content):
        """保存报告"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 报告已保存: {filepath}")

if __name__ == "__main__":
    print("LSTM + 随机森林混合模型 - 配置工具\n")

    # 尝试复制数据
    if DataLocator.copy_data_to_models():
        print("\n✅ 数据准备完成!")
    else:
        print("\n⚠️  数据准备失败，请确保数据文件存在")
