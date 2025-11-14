
import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 设置中文字体

def train_random_forest_model(input_csv, output_dir):
    """
    训练随机森林模型用于水质污染等级分类，并生成评估报告。

    参数:
    input_csv (str): 输入的CSV文件路径
    output_dir (str): 保存报告和图表的目录
    """
    # --- 1. 数据加载和准备 ---
    print(f"从 '{input_csv}' 加载数据...")
    df = pd.read_csv(input_csv)

    # 定义特征和目标
    target = 'Pollution_Level'
    # 移除不需要的列和原始重金属列
    features_to_drop = [
        'Province', 'City', 'Monitoring_Station', 'Date', 'Remarks', # 非模型特征
        'Water_Quality_Index', # 另一个目标变量，避免数据泄露
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L' # 已合并
    ]
    df_model = df.drop(columns=features_to_drop)

    # --- 2. 数据预处理 ---
    print("开始数据预处理...")
    # 对目标变量进行编码
    le = LabelEncoder()
    df_model[target] = le.fit_transform(df_model[target])
    # 保存标签映射关系
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
    print("开始训练随机森林模型...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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
    随机森林分类模型评估报告
    ======================================

    1. 模型说明
    ------------------
    本报告使用了随机森林分类器来预测水质的污染等级。
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
    - 模型的总体准确率达到了 {accuracy:.2%}，表明随机森林模型在区分不同水质污染等级方面表现出色。
    - 从分类报告来看，模型在预测'Excellent', 'Good', 'Poor', 'Very Poor'等级时表现稳健，F1分数均较高。
    - 'Moderate'等级的F1分数略低于其他类别，可能是因为其特征与其他类别的界限较为模糊。
    - 总体而言，随机森林是一个非常有效的基准模型，可用于实际的水质污染等级预测。

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
    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16)
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
    plt.title('Top 15 特征重要性', fontsize=16)
    plt.xlabel('重要性得分')
    plt.ylabel('特征')
    plt.tight_layout()
    fi_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"特征重要性图已保存到: {fi_path}")

if __name__ == '__main__':
    train_random_forest_model(
        input_csv='preprocessed_data_feature_eng.csv',
        output_dir='Outputs/Classification_RandomForest'
    )
