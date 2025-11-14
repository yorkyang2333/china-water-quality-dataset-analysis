
import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 设置中文字体，确保图表能正确显示中文

def create_new_features(df):
    """
    根据现有数据创建新的组合特征

    参数:
    df (pd.DataFrame): 包含预处理数据的DataFrame

    返回:
    pd.DataFrame: 包含新特征的DataFrame
    """
    # 1. 合并重金属指标
    # 创建一个总重金属指标，综合评估重金属污染
    heavy_metal_cols = ['Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L']
    df['Total_Heavy_Metals_ug_L'] = df[heavy_metal_cols].sum(axis=1)

    # 2. 创建有机污染指标比率 (COD/BOD)
    # 这个比率可以反映污水中难生物降解有机物的比例
    # 避免除以零的情况
    df['COD_BOD_Ratio'] = df['COD_mg_L'] / df['BOD_mg_L'].replace(0, np.nan)

    # 3. 创建氮磷比 (Total Nitrogen / Total Phosphorus)
    # 这是评估水体富营养化的重要指标
    # 避免除以零的情况
    df['NP_Ratio'] = df['Total_Nitrogen_mg_L'] / df['Total_Phosphorus_mg_L'].replace(0, np.nan)
    
    print("已成功创建新的组合特征：")
    print("- Total_Heavy_Metals_ug_L")
    print("- COD_BOD_Ratio")
    print("- NP_Ratio")
    
    return df

def analyze_new_features(df, output_dir):
    """
    分析新创建特征的有效性

    参数:
    df (pd.DataFrame): 包含新特征的DataFrame
    output_dir (str): 保存图表的目录路径
    """
    new_features = ['Total_Heavy_Metals_ug_L', 'COD_BOD_Ratio', 'NP_Ratio', 'Water_Quality_Index']
    
    # 1. 计算新特征与水质指数的相关性
    correlation_matrix = df[new_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('新特征与水质指数(WQI)的相关性分析', fontsize=16)
    plt.savefig(os.path.join(output_dir, '01_new_features_correlation_heatmap.png'), dpi=150)
    plt.close()
    print(f"图表 '01_new_features_correlation_heatmap.png' 已保存在 '{output_dir}'")

    # 2. 可视化新特征的分布
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(new_features[:-1], 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[feature].dropna(), kde=True)
        plt.title(f'{feature} 的分布', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_new_features_distribution.png'), dpi=150)
    plt.close()
    print(f"图表 '02_new_features_distribution.png' 已保存在 '{output_dir}'")

    # 3. 分析新特征与污染等级的关系
    plt.figure(figsize=(18, 6))
    for i, feature in enumerate(new_features[:-1], 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='Pollution_Level', y=feature, data=df, 
                    order=['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor'])
        plt.title(f'{feature} 在不同污染等级下的表现', fontsize=14)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_new_features_vs_pollution_level.png'), dpi=150)
    plt.close()
    print(f"图表 '03_new_features_vs_pollution_level.png' 已保存在 '{output_dir}'")

def main():
    # 定义输入和输出路径
    input_csv = 'preprocessed_data.csv'
    output_csv = 'preprocessed_data_feature_eng.csv'
    output_dir = 'Outputs/Feature_Engineering_EDA'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print(f"正在从 '{input_csv}' 加载数据...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"错误: 未找到输入文件 '{input_csv}'。请确保文件存在。")
        return
    
    # 创建新特征
    df_featured = create_new_features(df.copy())

    # 分析新特征
    print("\n开始分析新创建的特征...")
    analyze_new_features(df_featured, output_dir)

    # 保存包含新特征的数据集
    print(f"\n正在将包含新特征的数据保存到 '{output_csv}'...")
    df_featured.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("\nEDA与特征工程分析完成！")
    print(f"结果保存在 '{output_dir}' 和 '{output_csv}' 中。")

if __name__ == '__main__':
    main()
