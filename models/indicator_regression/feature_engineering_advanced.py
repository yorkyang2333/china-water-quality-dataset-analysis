"""
高级特征工程 - 针对失败指标的领域知识驱动特征
Advanced Feature Engineering for Failed Indicators (DO, NH3-N, pH)

作者: Claude Code
日期: 2025-11-13
用途: 为溶解氧、氨氮、pH创建物理化学特征，提升预测性能
"""

import pandas as pd
import numpy as np

def add_dissolved_oxygen_features(df):
    """
    添加溶解氧相关特征

    基于物理化学原理：
    - DO饱和浓度受温度影响（Weiss公式简化版）
    - DO饱和度反映水体氧化还原状态
    - DO/BOD比反映氧供需平衡
    """
    print("[FEATURE] 添加溶解氧相关特征...")

    # 1. DO饱和浓度（mg/L）- 温度依赖公式（淡水，标准大气压）
    # 公式来源：Weiss (1970) 简化版
    T = df['Water_Temperature_C']
    df['DO_saturation_mg_L'] = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3

    # 2. DO饱和度（百分比）
    df['DO_saturation_percent'] = (df['Dissolved_Oxygen_mg_L'] / df['DO_saturation_mg_L']) * 100

    # 3. DO亏损（mg/L）
    df['DO_deficit_mg_L'] = df['DO_saturation_mg_L'] - df['Dissolved_Oxygen_mg_L']

    # 4. DO/BOD比 - 氧供需比（避免除零）
    df['DO_BOD_ratio'] = df['Dissolved_Oxygen_mg_L'] / (df['BOD_mg_L'] + 1e-6)

    print(f"  [OK] 添加了4个溶解氧特征")
    return df

def add_nitrogen_cycle_features(df):
    """
    添加氮循环相关特征

    基于氮循环原理：
    - TIN（总无机氮）= NO3 + NO2 + NH3
    - 氨氮/总氮比反映硝化程度
    - 硝态氮/氨氮比反映氧化还原状态
    - 有机氮 = 总氮 - TIN
    """
    print("[FEATURE] 添加氮循环相关特征...")

    # 1. 总无机氮（TIN, mg/L）
    df['Total_Inorganic_N_mg_L'] = (
        df['Nitrate_mg_L'] +
        df['Nitrite_mg_L'] +
        df['Ammonia_N_mg_L']
    )

    # 2. 氨氮占总氮比例（0-1）- 反映硝化程度
    df['Ammonia_TN_ratio'] = df['Ammonia_N_mg_L'] / (df['Total_Nitrogen_mg_L'] + 1e-6)

    # 3. 硝态氮/氨氮比 - 氧化还原状态指标
    df['Nitrate_Ammonia_ratio'] = df['Nitrate_mg_L'] / (df['Ammonia_N_mg_L'] + 1e-6)

    # 4. 有机氮估算（mg/L）
    df['Organic_N_mg_L'] = df['Total_Nitrogen_mg_L'] - df['Total_Inorganic_N_mg_L']
    df['Organic_N_mg_L'] = df['Organic_N_mg_L'].clip(lower=0)  # 避免负值

    # 5. 氨氮对数变换（处理偏态分布）
    df['log_Ammonia_N'] = np.log1p(df['Ammonia_N_mg_L'])

    print(f"  [OK] 添加了5个氮循环特征")
    return df

def add_ph_buffer_features(df):
    """
    添加pH缓冲容量相关特征

    基于水化学原理：
    - 碱度（通过电导率估算）
    - 离子强度（影响pH缓冲容量）
    - 有机酸含量（难降解有机物）
    - H+浓度（pH的物理意义）
    """
    print("[FEATURE] 添加pH缓冲相关特征...")

    # 1. 碱度代理（mg/L as CaCO3）- 经验公式
    # 淡水中电导率与碱度的关系：Alkalinity ≈ 0.6 × Conductivity
    df['Alkalinity_proxy'] = df['Conductivity_uS_cm'] * 0.6

    # 2. 离子强度代理（mM）
    # 简化估算：I ≈ Conductivity / 1000
    df['Ionic_strength_proxy'] = df['Conductivity_uS_cm'] / 1000

    # 3. 有机酸代理（mg/L）- 难生物降解有机物
    # COD包含可生物降解（BOD）和难降解部分
    df['Organic_acid_proxy'] = df['COD_mg_L'] - df['BOD_mg_L']
    df['Organic_acid_proxy'] = df['Organic_acid_proxy'].clip(lower=0)

    # 4. H+浓度（mol/L）- pH的物理意义
    df['H_concentration'] = 10**(-df['pH'])

    print(f"  [OK] 添加了4个pH缓冲特征")
    return df

def add_redox_features(df):
    """
    添加氧化还原状态特征（通用）

    这些特征对DO、NH3-N、pH都有潜在预测价值
    """
    print("[FEATURE] 添加氧化还原状态特征...")

    # 1. 硝态氮/亚硝态氮比 - 氧化程度指标
    df['Nitrate_Nitrite_ratio'] = df['Nitrate_mg_L'] / (df['Nitrite_mg_L'] + 1e-6)

    # 2. DO/COD比 - 氧化能力指标
    df['DO_COD_ratio'] = df['Dissolved_Oxygen_mg_L'] / (df['COD_mg_L'] + 1e-6)

    print(f"  [OK] 添加了2个氧化还原特征")
    return df

def engineer_advanced_features(df):
    """
    统一入口：添加所有高级特征

    返回：
    - df: 添加了新特征的DataFrame
    - new_feature_names: 新增特征名称列表
    """
    print("\n" + "="*70)
    print("  高级特征工程 - 针对失败指标（DO/NH3-N/pH）")
    print("="*70)

    original_columns = df.columns.tolist()

    # 添加各类特征
    df = add_dissolved_oxygen_features(df)
    df = add_nitrogen_cycle_features(df)
    df = add_ph_buffer_features(df)
    df = add_redox_features(df)

    # 统计新增特征
    new_columns = [col for col in df.columns if col not in original_columns]

    print(f"\n[SUMMARY] 共添加 {len(new_columns)} 个新特征:")
    for i, col in enumerate(new_columns, 1):
        print(f"  {i}. {col}")

    print("="*70 + "\n")

    return df, new_columns

def validate_features(df, new_feature_names):
    """
    验证新特征的质量

    检查：
    - 缺失值比例
    - 异常值（inf, -inf）
    - 与目标指标的相关性
    """
    print("[VALIDATION] 验证新特征质量...")

    issues = []

    for col in new_feature_names:
        # 检查缺失值
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 5:
            issues.append(f"{col}: 缺失值 {missing_pct:.1f}%")

        # 检查无穷值
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            issues.append(f"{col}: {inf_count} 个无穷值")
            # 替换为NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    if issues:
        print("[WARN] 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[OK] 所有特征验证通过")

    return df

if __name__ == "__main__":
    # 测试功能
    print("\n[TEST] 测试特征工程模块...")

    # 加载数据
    data_path = "../../preprocessed_data_feature_eng.csv"
    df = pd.read_csv(data_path)
    print(f"[OK] 加载数据: {df.shape}")

    # 应用特征工程
    df, new_features = engineer_advanced_features(df)

    # 验证特征
    df = validate_features(df, new_features)

    # 计算与目标指标的相关性
    print("\n[CORRELATION] 新特征与目标指标的相关性:")
    targets = ['Dissolved_Oxygen_mg_L', 'Ammonia_N_mg_L', 'pH']

    for target in targets:
        print(f"\n{target}:")
        corr = df[new_features + [target]].corr()[target].drop(target).abs().sort_values(ascending=False).head(5)
        for feat, val in corr.items():
            print(f"  {feat}: r = {val:.4f}")

    # 保存扩展数据（可选）
    output_path = "../../preprocessed_data_advanced_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[OK] 保存扩展数据集: {output_path}")

    print("\n[DONE] 特征工程测试完成")


def get_clean_features():
    """
    返回无泄露高级特征列表（用于真实预测场景）

    这些特征不包含预测目标值或其变换，可安全用于：
    - 未来预测
    - 假设场景分析
    - 学术论文报告

    返回：
        list: 4个无泄露特征名称
    """
    return [
        'DO_saturation_mg_L',      # 仅依赖水温（无泄露）
        'Alkalinity_proxy',        # 仅依赖电导率（无泄露）
        'Ionic_strength_proxy',    # 仅依赖电导率（无泄露）
        'Organic_acid_proxy'       # COD - BOD（无泄露）
    ]


def get_leaky_features():
    """
    返回有泄露高级特征列表（仅用于缺失值填补场景）

    警告：这些特征包含预测目标值或其变换，不可用于未来预测！

    适用场景：
    - 同时刻缺失值填补（已知目标值的变换）
    - 传感器故障时的值估算

    返回：
        dict: {特征名: 泄露源}
    """
    return {
        'log_Ammonia_N': 'Ammonia_N_mg_L',  # 直接对数变换目标
        'DO_saturation_percent': 'Dissolved_Oxygen_mg_L',  # 包含DO目标
        'DO_deficit_mg_L': 'Dissolved_Oxygen_mg_L',  # 包含DO目标
        'DO_BOD_ratio': 'Dissolved_Oxygen_mg_L',  # 包含DO目标
        'Ammonia_TN_ratio': 'Ammonia_N_mg_L',  # 包含氨氮目标
        'Total_Inorganic_N_mg_L': 'Ammonia_N_mg_L',  # 包含氨氮目标
        'Nitrate_Ammonia_ratio': 'Ammonia_N_mg_L',  # 分母包含氨氮目标
        'Organic_N_mg_L': 'Ammonia_N_mg_L',  # 间接包含氨氮目标
        'H_concentration': 'pH',  # 直接变换pH目标
        'DO_COD_ratio': 'Dissolved_Oxygen_mg_L',  # 包含DO目标
        'Nitrate_Nitrite_ratio': None  # 无泄露（但依赖其他指标）
    }


def engineer_clean_features(df):
    """
    生成仅包含无泄露特征的数据集（用于学术研究和真实预测）

    参数：
        df: 输入数据框（必须包含基础指标）

    返回：
        df: 添加了4个无泄露特征的数据框
        features: 无泄露特征列表
    """
    print("\n[FEATURE ENGINEERING - CLEAN VERSION]")
    print("生成无泄露高级特征（仅4个）...")

    # 1. DO饱和度（Weiss公式）- 仅依赖水温
    T = df['Water_Temperature_C']
    df['DO_saturation_mg_L'] = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3
    print("  [OK] DO_saturation_mg_L (依赖: 水温)")

    # 2. 碱度代理 - 仅依赖电导率
    df['Alkalinity_proxy'] = df['Conductivity_uS_cm'] * 0.6
    print("  [OK] Alkalinity_proxy (依赖: 电导率)")

    # 3. 离子强度代理 - 仅依赖电导率
    df['Ionic_strength_proxy'] = df['Conductivity_uS_cm'] / 1000
    print("  [OK] Ionic_strength_proxy (依赖: 电导率)")

    # 4. 有机酸代理 - COD - BOD
    df['Organic_acid_proxy'] = df['COD_mg_L'] - df['BOD_mg_L']
    df['Organic_acid_proxy'] = df['Organic_acid_proxy'].clip(lower=0)
    print("  [OK] Organic_acid_proxy (依赖: COD, BOD)")

    clean_features = get_clean_features()
    print(f"\n[OK] 无泄露特征生成完成: {len(clean_features)} 个")
    print(f"    特征列表: {clean_features}")

    return df, clean_features
