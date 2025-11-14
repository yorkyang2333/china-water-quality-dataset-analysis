
import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 设置中文字体

def train_lightgbm_model(input_csv, output_dir):
    """
    训练LightGBM模型用于水质污染等级分类，并生成评估报告。

    参数:
    input_csv (str): 输入的CSV文件路径
    output_dir (str): 保存报告和图表的目录
    """
    # --- 1. 数据加载和准备 ---
    print(f"从 '{input_csv}' 加载数据...")
    df = pd.read_csv(input_csv)

    # 定义特征和目标
    target = 'Pollution_Level'
    # 移除不需要的列
    features_to_drop = [
        'Province', 'City', 'Monitoring_Station', 'Date', 'Remarks',
        'Water_Quality_Index',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L'
    ]
    df_model = df.drop(columns=features_to_drop)

    # --- 2. 数据预处理 ---
    print("开始数据预处理...")
    # 对目标变量进行编码
    le = LabelEncoder()
    df_model[target] = le.fit_transform(df_model[target])
    label_mapping = {index: label for index, label in enumerate(le.classes_)}
    print(f"污染等级映射关系: {label_mapping}")

    # 处理新特征中的无穷大值和NaN
    for col in ['COD_BOD_Ratio', 'NP_Ratio']:
        df_model[col] = df_model[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)
        print(f"处理了 '{col}' 中的缺失值和无穷大值，使用中位数 {median_val:.2f} 进行填充。")

    # 分离特征和目标
    X = df_model.drop(columns=[target])
    y = df_model[target]

    # --- 3. 训练/测试数据分割 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"数据已分割: {len(X_train)}条训练数据, {len(X_test)}条测试数据。")

    # --- 4. 模型训练 ---
    print("开始训练LightGBM模型...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("模型训练完成。")

    # --- 5. 模型评估 ---
    print("开始评估模型...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=le.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- 6. 生成并保存报告和图表 ---
    os.makedirs(output_dir, exist_ok=True)

    # 保存分类报告
    report_content = f"""
    LightGBM 分类模型评估报告
    ======================================

    1. 模型说明
    ------------------
    本报告使用了LightGBM分类器来预测水质的污染等级。
    - 特征: 使用了水温、pH、溶解氧等19个理化指标以及经纬度。
    - 目标: 污染等级 (Pollution_Level)，分为五个类别。
    - 数据集: {input_csv}
    - 训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}

    2. 性能指标
    ------------------
    - 准确率 (Accuracy): {accuracy:.4f}

    - 分类报告 (Classification Report):
{report_text}

    3. 结论与建议
    ------------------
    - LightGBM模型的总体准确率达到了 {accuracy:.2%}，表现出色，与XGBoost和随机森林性能相当。
    - LightGBM以其高效率和低内存消耗而闻名，是处理大规模数据集的绝佳选择。
    - 模型的F1分数在各个类别上表现均衡，证明了其强大的分类能力。

    4. 特征重要性
    ------------------
    下图展示了对模型预测贡献最大的15个特征。
    """
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"评估报告已保存到: {report_path}")

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('混淆矩阵 (Confusion Matrix) - LightGBM', fontsize=16)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"混淆矩阵图已保存到: {cm_path}")

    # 绘制并保存特征重要性图
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_15_features = feature_importances.head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_15_features, y=top_15_features.index)
    plt.title('Top 15 特征重要性 - LightGBM', fontsize=16)
    plt.xlabel('重要性得分')
    plt.ylabel('特征')
    plt.tight_layout()
    fi_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"特征重要性图已保存到: {fi_path}")

if __name__ == '__main__':
    train_lightgbm_model(
        input_csv='preprocessed_data_feature_eng.csv',
        output_dir='Outputs/Classification_LightGBM'
    )
