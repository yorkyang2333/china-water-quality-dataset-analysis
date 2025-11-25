"""
XGBoost污染等级分类 - 按站点分割版本（空间泛化评估）
XGBoost Classification with Spatial Cross-Validation (Leave-One-Station-Out)

目的：评估模型的空间泛化能力，避免随机分割导致的过拟合风险

修改记录（2025-11-25）：
- 实现按站点分割（Leave-Group-Out）策略
- 对比随机分割vs按站点分割的性能差异
- 根据审稿意见重新评估真实泛化能力

作者: Claude Code
版本: v2.0
日期: 2025-11-25
"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*80)
print("  XGBoost分类 - 空间泛化评估（两种分割策略对比）")
print("  XGBoost Classification with Spatial Generalization Assessment")
print("="*80)

# 创建输出目录
OUTPUT_DIR = "../../../Outputs/Classification_XGBoost_Spatial_Split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载数据
print("\n[STEP 1] 加载数据...")
df = pd.read_csv("../../../preprocessed_data_feature_eng.csv")
print(f"[OK] 数据加载完成: {df.shape}")

# 2. 数据预处理
print("\n[STEP 2] 数据预处理...")

# 定义特征和目标
target = 'Pollution_Level'

# 保留Monitoring_Station用于分组（后面会移除）
station_col = 'Monitoring_Station'

# 移除不需要的列（但保留Monitoring_Station用于分组）
features_to_drop = [
    'Province', 'City', 'Date', 'Remarks',
    'Water_Quality_Index',  # 排除WQI（会泄露目标）
    'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L'  # 使用Total_Heavy_Metals替代
]

# 创建模型数据集（暂时保留Monitoring_Station）
df_model = df.drop(columns=features_to_drop)

# 对目标变量进行编码
le = LabelEncoder()
df_model[target] = le.fit_transform(df_model[target])
label_mapping = {index: label for index, label in enumerate(le.classes_)}
print(f"污染等级映射: {label_mapping}")

# 处理工程特征中的无穷大值和NaN
for col in ['COD_BOD_Ratio', 'NP_Ratio']:
    if col in df_model.columns:
        df_model[col] = df_model[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)
        print(f"  处理 '{col}': 中位数填充 = {median_val:.2f}")

# 分离站点信息、特征和目标
groups = df_model[station_col]  # 用于分组
X_full = df_model.drop(columns=[target, station_col])  # 特征（不含站点名）
y_full = df_model[target]  # 目标

print(f"\n[INFO] 数据统计:")
print(f"  样本总数: {len(X_full)}")
print(f"  特征数: {X_full.shape[1]}")
print(f"  监测站数: {groups.nunique()}")
print(f"  各站点样本数分布: min={groups.value_counts().min()}, "
      f"max={groups.value_counts().max()}, "
      f"mean={groups.value_counts().mean():.1f}")

# 3. 定义训练评估函数
def train_and_evaluate(X_train, X_test, y_train, y_test, split_name):
    """训练XGBoost并评估"""
    print(f"\n{'='*80}")
    print(f"  {split_name}")
    print(f"{'='*80}")

    print(f"\n[TRAIN] 训练集: {len(X_train)} 样本")
    print(f"[TEST]  测试集: {len(X_test)} 样本")

    # 训练模型
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_mapping),
        use_label_encoder=False,
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算准确率
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n[RESULTS] 性能指标:")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")
    print(f"  准确率差异: {train_acc - test_acc:.4f}")

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # 分类报告
    report = classification_report(y_test, y_pred_test,
                                   target_names=le.classes_,
                                   output_dict=True)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': '样本数量'},
                annot_kws={'fontsize': 12})
    plt.xlabel('预测类别', fontsize=14, fontweight='bold')
    plt.ylabel('真实类别', fontsize=14, fontweight='bold')
    plt.title(f'混淆矩阵 - {split_name}\n准确率={test_acc:.4f}',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    filename = split_name.replace("（", "_").replace("）", "").replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{filename}.png", dpi=300, bbox_inches='tight')
    print(f"  [OK] 混淆矩阵已保存")
    plt.close()

    return {
        'split_name': split_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'acc_diff': train_acc - test_acc,
        'conf_matrix': conf_matrix,
        'report': report,
        'model': model
    }


# 4. 策略1：随机分割（传统方法，基线）
print("\n" + "="*80)
print("  策略1：随机分割（Random Split）")
print("  说明：训练集和测试集可能包含同一站点的数据")
print("="*80)

X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

results_random = train_and_evaluate(
    X_train_random, X_test_random,
    y_train_random, y_test_random,
    "随机分割（Random Split）"
)


# 5. 策略2：按站点分割（空间泛化评估）
print("\n" + "="*80)
print("  策略2：按站点分割（Spatial Split - Leave-Group-Out）")
print("  说明：训练集和测试集包含完全不同的监测站")
print("="*80)

# 使用GroupShuffleSplit确保同一站点的数据不会同时出现在训练和测试集
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_full, y_full, groups=groups))

X_train_spatial = X_full.iloc[train_idx]
X_test_spatial = X_full.iloc[test_idx]
y_train_spatial = y_full.iloc[train_idx]
y_test_spatial = y_full.iloc[test_idx]
groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

# 验证分割正确性
assert len(set(groups_train) & set(groups_test)) == 0, "错误：训练和测试集有重叠站点！"
print(f"[OK] 分割验证通过:")
print(f"  训练集站点数: {groups_train.nunique()}")
print(f"  测试集站点数: {groups_test.nunique()}")
print(f"  站点重叠数: 0 (验证通过)")

results_spatial = train_and_evaluate(
    X_train_spatial, X_test_spatial,
    y_train_spatial, y_test_spatial,
    "按站点分割（Spatial Split）"
)


# 6. 对比分析
print("\n" + "="*80)
print("  对比分析：随机分割 vs 按站点分割")
print("="*80)

comparison = pd.DataFrame([
    {
        '分割策略': '随机分割',
        '训练集准确率': results_random['train_acc'],
        '测试集准确率': results_random['test_acc'],
        '准确率差异': results_random['acc_diff'],
        '说明': '可能高估泛化能力（站点重叠）'
    },
    {
        '分割策略': '按站点分割',
        '训练集准确率': results_spatial['train_acc'],
        '测试集准确率': results_spatial['test_acc'],
        '准确率差异': results_spatial['acc_diff'],
        '说明': '真实空间泛化能力'
    }
])

print("\n表：两种分割策略性能对比\n")
print(comparison.to_string(index=False))

# 计算性能下降
acc_drop = results_random['test_acc'] - results_spatial['test_acc']
acc_drop_pct = (acc_drop / results_random['test_acc']) * 100

print(f"\n[ANALYSIS] 空间泛化性能下降:")
print(f"  绝对下降: {acc_drop:.4f} ({acc_drop*100:.2f} 个百分点)")
print(f"  相对下降: {acc_drop_pct:.2f}%")

if acc_drop > 0.05:
    print(f"  [WARN] 准确率下降超过5个百分点，表明随机分割可能高估了模型的泛化能力")
    print(f"  [WARN] 模型可能学习到了站点特定的模式而非通用的水质-污染关系")
else:
    print(f"  [OK] 准确率下降较小，模型具有良好的空间泛化能力")

# 保存对比结果
comparison.to_csv(f"{OUTPUT_DIR}/comparison_results.csv", index=False, encoding='utf-8-sig')
print(f"\n[OK] 对比结果已保存: {OUTPUT_DIR}/comparison_results.csv")

# 7. 生成详细报告
report_content = f"""
XGBoost污染等级分类 - 空间泛化评估报告
================================================================================

一、实验目的
  评估XGBoost分类模型在两种数据分割策略下的性能差异，识别随机分割可能
  导致的泛化能力高估问题。

二、数据统计
  - 样本总数: {len(X_full)}
  - 特征数: {X_full.shape[1]}
  - 污染等级: {len(label_mapping)} 类 {list(label_mapping.values())}
  - 监测站数: {groups.nunique()}
  - 训练/测试比例: 80/20

三、两种分割策略

3.1 随机分割（Random Split）
  - 方法: sklearn.model_selection.train_test_split
  - 特点: 同一站点的数据可能同时出现在训练集和测试集
  - 风险: 可能导致模型学习站点特定模式，高估泛化能力

  性能指标:
    训练集准确率: {results_random['train_acc']:.4f}
    测试集准确率: {results_random['test_acc']:.4f}
    准确率差异: {results_random['acc_diff']:.4f}

3.2 按站点分割（Spatial Split）
  - 方法: sklearn.model_selection.GroupShuffleSplit
  - 特点: 训练集和测试集包含完全不同的监测站
  - 优势: 评估真实的空间泛化能力

  性能指标:
    训练集准确率: {results_spatial['train_acc']:.4f}
    测试集准确率: {results_spatial['test_acc']:.4f}
    准确率差异: {results_spatial['acc_diff']:.4f}

  分割验证:
    训练集站点数: {groups_train.nunique()}
    测试集站点数: {groups_test.nunique()}
    站点重叠: 0 (验证通过)

四、对比分析

4.1 准确率对比
  随机分割测试集准确率: {results_random['test_acc']:.4f}
  按站点分割测试集准确率: {results_spatial['test_acc']:.4f}
  绝对下降: {acc_drop:.4f} ({acc_drop*100:.2f} 个百分点)
  相对下降: {acc_drop_pct:.2f}%

4.2 结论
  {'随机分割显著高估了模型的泛化能力（下降>5%），模型可能学习到了站点特定的空间模式。' if acc_drop > 0.05 else '两种分割策略的性能接近，模型具有良好的空间泛化能力。'}

五、建议
  1. 论文中应报告按站点分割的结果作为真实泛化能力的估计
  2. 随机分割结果可作为对比基线，但需说明其局限性
  3. 如需提升空间泛化能力，可考虑：
     - 增加地理位置特征（经纬度）
     - 使用更多监测站的数据
     - 引入迁移学习或域自适应方法

六、输出文件
  - 混淆矩阵（随机分割）: confusion_matrix_Random_Split.png
  - 混淆矩阵（按站点分割）: confusion_matrix_Spatial_Split.png
  - 对比结果表: comparison_results.csv
  - 本报告: spatial_generalization_report.txt

================================================================================
生成时间: 2025-11-25
版本: v2.0
================================================================================
"""

with open(f"{OUTPUT_DIR}/spatial_generalization_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n[OK] 详细报告已保存: {OUTPUT_DIR}/spatial_generalization_report.txt")

print("\n" + "="*80)
print("  实验完成！")
print(f"  输出目录: {OUTPUT_DIR}")
print("="*80)
