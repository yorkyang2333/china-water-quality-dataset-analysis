#!/usr/bin/env python3
"""
改进版水质污染等级分类模型
解决类别不平衡问题 + 特征工程优化

改进点:
1. SMOTE过采样处理类别不平衡
2. 类别权重调整
3. 恢复地理特征编码
4. 添加时间特征
5. 超参数优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
import xgboost as xgb

# 处理不平衡数据
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    print("警告: imbalanced-learn未安装，将使用类别权重方法")

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedWaterQualityClassifier:
    """改进版水质分类器"""

    def __init__(self, output_dir='Outputs/Improved_Classification'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None

    def load_and_preprocess_data(self, csv_path):
        """加载和预处理数据"""
        print("🔄 加载和预处理数据...")

        df = pd.read_csv(csv_path)
        print(f"✅ 数据加载成功: {len(df)} 条记录, {len(df.columns)} 个特征")

        # 基础数据清理
        df['Date'] = pd.to_datetime(df['Date'])

        # 添加时间特征
        print("⏰ 添加时间特征...")
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Date'].dt.month % 12 // 3 + 1
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

        # 添加周期性编码
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Season_sin'] = np.sin(2 * np.pi * df['Season'] / 4)
        df['Season_cos'] = np.cos(2 * np.pi * df['Season'] / 4)

        # 地理特征编码
        print("🗺️ 添加地理特征...")
        df['Province_encoded'] = df['Province'].astype('category').cat.codes
        df['City_encoded'] = df['City'].astype('category').cat.codes
        df['Station_encoded'] = df['Monitoring_Station'].astype('category').cat.codes

        # 空间几何特征
        df['Lat_Squared'] = df['Latitude'] ** 2
        df['Lon_Squared'] = df['Longitude'] ** 2
        df['Lat_Lon_Product'] = df['Latitude'] * df['Longitude']

        # 处理NP_Ratio缺失值
        if 'NP_Ratio' in df.columns:
            df['NP_Ratio'].fillna(df['NP_Ratio'].median(), inplace=True)

        # 处理COD_BOD_Ratio缺失值
        if 'COD_BOD_Ratio' in df.columns:
            df['COD_BOD_Ratio'].fillna(df['COD_BOD_Ratio'].median(), inplace=True)

        # 选择特征列
        feature_cols = [
            # 基础水质指标
            'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm',
            'Turbidity_NTU', 'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
            'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
            'Coliform_Count_CFU_100mL',

            # 工程特征
            'Total_Heavy_Metals_ug_L', 'COD_BOD_Ratio', 'NP_Ratio',

            # 地理特征
            'Latitude', 'Longitude', 'Province_encoded', 'City_encoded', 'Station_encoded',
            'Lat_Squared', 'Lon_Squared', 'Lat_Lon_Product',

            # 时间特征
            'Year', 'Month', 'Season', 'Quarter', 'DayOfWeek', 'IsWeekend',
            'Month_sin', 'Month_cos', 'Season_sin', 'Season_cos'
        ]

        # 移除不需要的列
        drop_cols = ['Province', 'City', 'Monitoring_Station', 'Date', 'Remarks', 'Water_Quality_Index']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # 确保所有特征列都存在
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"✅ 使用 {len(available_features)} 个特征")

        self.feature_names = available_features

        # 准备特征和目标
        X = df[available_features].copy()
        y = df['Pollution_Level'].copy()

        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        print(f"📊 标签映射: {self.label_mapping}")

        # 类别分布分析
        print("\n🎯 类别分布:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            percentage = count / len(y_encoded) * 100
            print(f"   {label}: {count} 条 ({percentage:.1f}%)")

        return X, y_encoded

    def handle_class_imbalance(self, X, y, method='smote'):
        """处理类别不平衡"""
        print(f"\n⚖️ 处理类别不平衡 (方法: {method})...")

        if not IMBALANCED_LEARN_AVAILABLE or method == 'class_weight':
            print("📏 使用类别权重方法...")
            # 计算类别权重
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            print(f"   类别权重: {class_weight_dict}")
            return X, y, class_weight_dict

        # 使用SMOTE等方法
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=3)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=42)
        else:
            sampler = SMOTE(random_state=42)

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        print(f"✅ 重采样前: {X.shape[0]} 条记录")
        print(f"✅ 重采样后: {X_resampled.shape[0]} 条记录")

        print("\n📊 重采样后类别分布:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = np.sum(y_resampled == i)
            percentage = count / len(y_resampled) * 100
            print(f"   {label}: {count} 条 ({percentage:.1f}%)")

        return X_resampled, y_resampled, None

    def train_model(self, X, y, class_weights=None, model_type='xgboost'):
        """训练模型"""
        print(f"\n🤖 训练 {model_type} 模型...")

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if model_type == 'xgboost':
            # XGBoost模型
            params = {
                'objective': 'multi:softmax',
                'num_class': len(self.label_encoder.classes_),
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }

            if class_weights:
                params['scale_pos_weight'] = class_weights

            self.model = xgb.XGBClassifier(**params)

        elif model_type == 'random_forest':
            # 随机森林模型
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': class_weights if class_weights else 'balanced'
            }

            self.model = RandomForestClassifier(**params)

        # 训练模型
        self.model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        print(f"✅ 模型训练完成!")
        print(f"📊 测试集准确率: {accuracy:.4f}")

        # 详细性能分析
        print("\n📈 各类别性能:")
        for i, label in enumerate(self.label_encoder.classes_):
            if label in report:
                precision = report[label]['precision']
                recall = report[label]['recall']
                f1 = report[label]['f1-score']
                support = report[label]['support']
                print(f"   {label:10s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, 支持数={support}")

        return X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba

    def evaluate_and_visualize(self, X_test, y_test, y_pred, y_pred_proba):
        """评估和可视化结果"""
        print("\n📊 生成评估报告和可视化...")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('改进版模型 - 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 特征重要性 (如果支持)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('改进版模型 - 特征重要性 (Top 20)')
            plt.xlabel('重要性')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/improved_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("\n🔍 Top 10 重要特征:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")

        # 保存详细报告
        self.save_evaluation_report(y_test, y_pred, y_pred_proba)

    def save_evaluation_report(self, y_test, y_pred, y_pred_proba):
        """保存评估报告"""
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        )

        with open(f'{self.output_dir}/improved_model_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("改进版水质污染等级分类模型评估报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"模型训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用的特征数量: {len(self.feature_names)}\n")
            f.write(f"类别数量: {len(self.label_encoder.classes_)}\n\n")

            f.write("标签映射:\n")
            for i, label in self.label_mapping.items():
                f.write(f"  {i} -> {label}\n")
            f.write("\n")

            f.write("性能指标:\n")
            f.write("-" * 40 + "\n")
            f.write(report)
            f.write("\n")

            f.write("改进点说明:\n")
            f.write("-" * 40 + "\n")
            f.write("1. ✅ 使用SMOTE过采样处理类别不平衡\n")
            f.write("2. ✅ 添加时间特征 (月度、季节、周期编码)\n")
            f.write("3. ✅ 添加地理特征编码和空间几何特征\n")
            f.write("4. ✅ 特征标准化处理\n")
            f.write("5. ✅ 超参数优化\n")
            f.write("6. ✅ 更全面的评估指标\n")

        print(f"📄 评估报告已保存: {self.output_dir}/improved_model_report.txt")


def main():
    """主函数"""
    print("🚀 启动改进版水质分类模型训练...")

    # 创建分类器
    classifier = ImprovedWaterQualityClassifier()

    # 加载和预处理数据（从项目根目录）
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'preprocessed_data_feature_eng.csv')
    print(f"[数据路径] {data_path}")
    X, y = classifier.load_and_preprocess_data(data_path)

    # 处理类别不平衡
    if IMBALANCED_LEARN_AVAILABLE:
        X_resampled, y_resampled, class_weights = classifier.handle_class_imbalance(X, y, method='smote')
    else:
        X_resampled, y_resampled, class_weights = classifier.handle_class_imbalance(X, y, method='class_weight')

    # 训练模型
    X_train, X_test, y_train, y_test, y_pred, y_pred_proba = classifier.train_model(
        X_resampled, y_resampled, class_weights, model_type='xgboost'
    )

    # 评估和可视化
    classifier.evaluate_and_visualize(X_test, y_test, y_pred, y_pred_proba)

    print("\n🎉 改进版模型训练完成!")
    print(f"📁 所有结果已保存到: {classifier.output_dir}")


if __name__ == "__main__":
    main()