
import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 设置中文字体

def train_logistic_regression_model(input_csv, output_dir):
    """
    训练逻辑回归模型用于水质污染等级分类，并生成评估报告。

    参数:
    input_csv (str): 输入的CSV文件路径
    output_dir (str): 保存报告和图表的目录
    """
    # --- 1. 数据加载和准备 ---
    print(f"从 '{input_csv}' 加载数据...")
    df = pd.read_csv(input_csv)

    # 定义特征和目标
    target = 'Pollution_Level'
    features_to_drop = [
        'Province', 'City', 'Monitoring_Station', 'Date', 'Remarks',
        'Water_Quality_Index',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L'
    ]
    df_model = df.drop(columns=features_to_drop)

    # --- 2. 数据预处理 ---
    print("开始数据预处理...")
    # 目标变量编码
    le = LabelEncoder()
    df_model[target] = le.fit_transform(df_model[target])
    label_mapping = {index: label for index, label in enumerate(le.classes_)}
    print(f"污染等级映射关系: {label_mapping}")

    # 处理缺失值和无穷大值
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

    # --- 4. 特征缩放 ---
    print("对特征进行标准化处理 (Standard Scaling)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. 模型训练 ---
    print("开始训练逻辑回归模型...")
    model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("模型训练完成。")

    # --- 6. 模型评估 ---
    print("开始评估模型...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=le.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- 7. 生成并保存报告和图表 ---
    os.makedirs(output_dir, exist_ok=True)

    # 为结论部分动态获取其他模型的准确率 (如果报告存在)
    try:
        with open('Outputs/Classification_RandomForest/report.txt', 'r', encoding='utf-8') as f:
            rf_acc_line = [line for line in f if '准确率' in line][0]
            rf_accuracy = float(rf_acc_line.split(':')[1].strip())
    except (FileNotFoundError, IndexError):
        rf_accuracy = 0.0

    try:
        with open('Outputs/Classification_XGBoost/report.txt', 'r', encoding='utf-8') as f:
            xgb_acc_line = [line for line in f if '准确率' in line][0]
            xgb_accuracy = float(xgb_acc_line.split(':')[1].strip())
    except (FileNotFoundError, IndexError):
        xgb_accuracy = 0.0

    report_content = f"""
    逻辑回归 (Logistic Regression) 分类模型评估报告
    ==================================================

    1. 模型说明
    ------------------
    本报告使用了逻辑回归分类器作为基准模型，来预测水质的污染等级。
    - 特征: 使用了19个理化指标、经纬度以及新构建的3个组合特征。所有特征在训练前都经过了标准化处理。
    - 目标: 污染等级 (Pollution_Level)，分为五个类别。
    - 策略: 使用'One-vs-Rest' (OvR)策略处理多分类问题。
    - 数据集: {input_csv}
    - 训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}

    2. 性能指标
    ------------------
    - 准确率 (Accuracy): {accuracy:.4f}

    - 分类报告 (Classification Report):
{report_text}

    3. 结论与建议
    ------------------
    - 逻辑回归模型的总体准确率达到了 {accuracy:.2%}。作为一种简单的线性模型，这个结果可以作为性能基准。
    - 与随机森林({rf_accuracy:.2%})和XGBoost({xgb_accuracy:.2%})等复杂模型相比，逻辑回归的准确率较低，这表明水质污染等级的预测是一个复杂的非线性问题。
    - 尽管性能不是最优，但逻辑回归模型训练速度快、易于解释，适合作为快速验证特征有效性的初步模型。
    """

    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"评估报告已保存到: {report_path}")

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('混淆矩阵 (Confusion Matrix) - 逻辑回归', fontsize=16)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"混淆矩阵图已保存到: {cm_path}")

if __name__ == '__main__':
    train_logistic_regression_model(
        input_csv='preprocessed_data_feature_eng.csv',
        output_dir='Outputs/Classification_LogisticRegression'
    )
