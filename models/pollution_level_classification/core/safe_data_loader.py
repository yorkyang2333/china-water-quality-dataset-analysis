"""
安全的数据加载模块
防止数据泄露，严格遵循正确的数据分割顺序
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime


class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self.logs = []

    def log(self, step, description, details=None):
        """记录步骤"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'description': description,
            'details': details or {}
        }
        self.logs.append(entry)
        print(f"[审计] {step}: {description}")

    def get_report(self):
        """获取报告"""
        return self.logs


class WaterQualityDataLoader:
    """
    安全的水质数据加载器

    关键特性：
    1. 自动定位项目根目录
    2. 严格的数据分割顺序（分割 -> 统计 -> 平衡）
    3. 完整的审计日志
    4. 数据泄露检测支持
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.label_mapping = None
        self.scaler = StandardScaler()
        self.audit_log = AuditLogger()

        # 记录初始化
        self.audit_log.log(
            'INIT',
            '初始化数据加载器',
            {'test_size': test_size, 'random_state': random_state}
        )

    def get_project_root(self):
        """获取项目根目录（从core子目录向上4层）"""
        current_file = os.path.abspath(__file__)
        # core/safe_data_loader.py -> core -> pollution_level_classification -> models -> 项目根
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

    def load_raw_data(self):
        """
        [步骤1] 加载原始数据
        不做任何处理，仅加载
        """
        project_root = self.get_project_root()
        csv_path = os.path.join(project_root, 'preprocessed_data_feature_eng.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")

        self.audit_log.log(
            'STEP_1_LOAD',
            '加载原始数据文件',
            {'path': csv_path}
        )

        df = pd.read_csv(csv_path)
        self.audit_log.log(
            'STEP_1_LOAD',
            f'数据加载完成: {df.shape[0]} 行，{df.shape[1]} 列',
            {'shape': df.shape}
        )

        return df

    def basic_cleanup(self, df):
        """
        [步骤2] 基本清理（时间/空间清理，无目标信息泄露）

        注意：不使用预测目标的任何信息
        """
        self.audit_log.log('STEP_2_CLEANUP', '开始基本数据清理')

        df_clean = df.copy()

        # 移除完全重复的行
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]

        if removed > 0:
            self.audit_log.log(
                'STEP_2_CLEANUP',
                f'移除了 {removed} 个重复行'
            )

        # 检查缺失值
        missing = df_clean.isnull().sum()
        if missing.any():
            self.audit_log.log(
                'STEP_2_CLEANUP',
                '检测到缺失值',
                {'missing_counts': missing[missing > 0].to_dict()}
            )

        return df_clean

    def split_data(self, df):
        """
        [步骤3] 数据分割（Train/Test）

        ⚠️ 关键：所有后续操作必须基于Train数据的统计量！
        """
        self.audit_log.log(
            'STEP_3_SPLIT',
            '开始数据分割',
            {'test_size': self.test_size}
        )

        # 定义特征和目标
        target = 'Pollution_Level'
        features_to_drop = [
            'Province', 'City', 'Monitoring_Station', 'Date', 'Remarks',
            'Water_Quality_Index',
            'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L'
        ]

        df_model = df.drop(columns=features_to_drop)

        # 编码目标变量
        df_model[target] = self.label_encoder.fit_transform(df_model[target])
        self.label_mapping = {
            index: label for index, label in enumerate(self.label_encoder.classes_)
        }

        self.audit_log.log(
            'STEP_3_SPLIT',
            f'目标变量编码完成：{self.label_mapping}'
        )

        # 分割数据
        X = df_model.drop(columns=[target])
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.audit_log.log(
            'STEP_3_SPLIT',
            f'数据分割完成：Train={X_train.shape[0]}, Test={X_test.shape[0]}'
        )

        return X_train, X_test, y_train, y_test

    def compute_train_statistics(self, X_train):
        """
        [步骤4] 计算Train数据的统计量

        ✅ 所有后续的标准化/填补操作都使用这些统计量
        """
        self.audit_log.log(
            'STEP_4_TRAIN_STATS',
            '计算Train数据的统计量'
        )

        stats = {
            'mean': X_train.mean(),
            'std': X_train.std(),
            'median': X_train.median(),
            'min': X_train.min(),
            'max': X_train.max()
        }

        self.audit_log.log(
            'STEP_4_TRAIN_STATS',
            '统计量计算完成'
        )

        return stats

    def handle_missing_and_inf(self, X_train, X_test, train_stats):
        """
        [步骤5] 处理缺失值和无穷大

        ✅ 使用Train统计量处理所有数据
        """
        self.audit_log.log(
            'STEP_5_MISSING',
            '处理缺失值和无穷大'
        )

        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()

        # 特殊处理某些特征
        for col in ['COD_BOD_Ratio', 'NP_Ratio']:
            if col in X_train_clean.columns:
                # Train
                X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf], np.nan)
                X_train_clean[col].fillna(train_stats['median'][col], inplace=True)

                # Test - 使用Train的median
                X_test_clean[col] = X_test_clean[col].replace([np.inf, -np.inf], np.nan)
                X_test_clean[col].fillna(train_stats['median'][col], inplace=True)

        # 通用处理
        X_train_clean.fillna(X_train_clean.mean(), inplace=True)
        X_test_clean.fillna(X_train_clean.mean(), inplace=True)  # ✅ 使用Train的mean

        self.audit_log.log(
            'STEP_5_MISSING',
            '缺失值处理完成'
        )

        return X_train_clean, X_test_clean

    def standardize(self, X_train, X_test):
        """
        [步骤6] 标准化

        ✅ Scaler只在Train上fit，然后应用到Test
        """
        self.audit_log.log(
            'STEP_6_STANDARDIZE',
            '开始标准化处理'
        )

        self.scaler.fit(X_train)  # ✅ 只在Train上fit
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)  # ✅ 使用Train的参数

        # 转换回DataFrame保留列名
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        self.audit_log.log(
            'STEP_6_STANDARDIZE',
            '标准化完成'
        )

        return X_train_scaled, X_test_scaled

    def apply_smote(self, X_train, y_train, random_state=None):
        """
        [步骤7] 应用SMOTE（仅限Train数据）

        ⚠️ CRITICAL：SMOTE只应用于Train，绝不应用于Test！
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            self.audit_log.log(
                'STEP_7_SMOTE',
                '⚠️ imblearn未安装，跳过SMOTE',
                None
            )
            return X_train, y_train

        self.audit_log.log(
            'STEP_7_SMOTE',
            '开始应用SMOTE',
            {
                'before': X_train.shape,
                'class_distribution': y_train.value_counts().to_dict()
            }
        )

        smote = SMOTE(random_state=random_state or self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        self.audit_log.log(
            'STEP_7_SMOTE',
            'SMOTE应用完成',
            {
                'after': X_train_balanced.shape,
                'class_distribution': pd.Series(y_train_balanced).value_counts().to_dict()
            }
        )

        return X_train_balanced, y_train_balanced

    def preprocess(self, balance_method=None, apply_smote=False):
        """
        完整的预处理流程

        ✅ 严格顺序：加载 -> 清理 -> 分割 -> 统计 -> 处理缺失 -> 标准化 -> 平衡
        """
        self.audit_log.log(
            'PREPROCESS',
            '='*60 + ' 开始完整预处理流程 ' + '='*60
        )

        # 步骤1: 加载
        df = self.load_raw_data()

        # 步骤2: 清理
        df_clean = self.basic_cleanup(df)

        # 步骤3: 分割
        X_train, X_test, y_train, y_test = self.split_data(df_clean)

        # 步骤4: 计算Train统计量
        train_stats = self.compute_train_statistics(X_train)

        # 步骤5: 处理缺失和无穷大
        X_train, X_test = self.handle_missing_and_inf(X_train, X_test, train_stats)

        # 步骤6: 标准化
        X_train, X_test = self.standardize(X_train, X_test)

        # 步骤7: 可选的SMOTE（仅Train）
        if apply_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)

        self.audit_log.log(
            'PREPROCESS',
            '='*60 + ' 预处理流程完成 ' + '='*60
        )

        return X_train, X_test, y_train, y_test, self.label_mapping

    def get_audit_report(self):
        """获取审计报告"""
        return self.audit_log.get_report()

    def save_audit_log(self, filepath):
        """保存审计日志"""
        import json

        logs = self.audit_log.get_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        print(f"[审计] 日志已保存至: {filepath}")
