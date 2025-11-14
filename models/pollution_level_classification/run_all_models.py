"""
批量运行所有分类模型的脚本

功能：
1. 加载数据
2. 运行数据泄露检测
3. 批量训练所有模型
4. 批量评估所有模型
5. 生成完整报告
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径，方便导入
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from core.safe_data_loader import WaterQualityDataLoader
    from core.leak_detector import DataLeakDetector, IntegrityChecker
    from model_hub import ModelHub
except ImportError as e:
    print(f"[错误] 导入失败: {e}")
    print(f"[信息] 尝试从 {current_dir} 导入")
    sys.exit(1)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("水质污染分类模型 - 批量运行脚本")
    print("="*70 + "\n")

    # ========== 步骤1: 加载数据 ==========
    print("[步骤1] 加载和预处理数据...")
    print("-" * 70)

    loader = WaterQualityDataLoader(test_size=0.2, random_state=42)

    try:
        _ = loader.load_raw_data()  # 加载但不使用（预处理会重新加载）
        X_train, X_test, y_train, y_test, label_mapping = loader.preprocess(
            apply_smote=False  # 先不用SMOTE，让各模型自己决定
        )

        # 保存审计日志
        audit_log_path = os.path.join(
            loader.get_project_root(),
            'Outputs', 'Pollution_Classification', 'audit_log.json'
        )
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        loader.save_audit_log(audit_log_path)

        print(f"[数据] Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"[标签] {label_mapping}\n")

    except Exception as e:
        print(f"[错误] 数据加载失败: {e}\n")
        return

    # ========== 步骤2: 数据泄露检测 ==========
    print("[步骤2] 数据泄露检测...")
    print("-" * 70)

    leak_detector = DataLeakDetector(correlation_threshold=0.7, verbose=True)
    _ = leak_detector.check_correlation_anomaly(
        X_test, y_test, X_test.columns
    )
    leak_detector.check_feature_names(X_train)
    leak_detector.check_data_distribution(X_train, X_test)
    leak_detector.check_class_distribution(y_train, y_test)

    # 保存泄露检测报告
    leak_report_path = os.path.join(
        loader.get_project_root(),
        'Outputs', 'Pollution_Classification', 'leak_detection_report.json'
    )
    leak_detector.save_report(leak_report_path)

    # 数据完整性检查
    integrity_checker = IntegrityChecker(verbose=True)
    integrity_checker.check_no_nan(X_train, 'Train')
    integrity_checker.check_no_nan(X_test, 'Test')
    integrity_checker.check_shape(X_train, X_test)
    integrity_checker.check_split_ratio(X_train.shape[0], X_test.shape[0])
    integrity_checker.print_summary()

    print()

    # ========== 步骤3: 创建ModelHub并注册模型 ==========
    print("[步骤3] 注册模型...")
    print("-" * 70)

    hub = ModelHub()

    # 注意：这里是示例框架，实际需要导入和注册真实的模型
    # 示例代码（需要根据实际模型进行修改）
    print("""
    [信息] 需要手动注册模型，示例代码：

    from traditional.xgboost_classifier import XGBoostClassifier
    from traditional.random_forest_classifier import RandomForestClassifier

    hub.register_model('XGBoost', XGBoostClassifier(), category='traditional')
    hub.register_model('RandomForest', RandomForestClassifier(), category='traditional')
    hub.register_model('DeepLearning', DeepLearningClassifier(), category='deep_learning')
    # ... 注册其他模型
    """)

    print(f"[模型] 已注册 {len(hub.get_all_models())} 个模型\n")

    if len(hub.get_all_models()) == 0:
        print("[警告] 未注册任何模型，请手动注册模型后重新运行\n")
        return

    # ========== 步骤4: 批量训练 ==========
    print("[步骤4] 批量训练模型...")
    print("-" * 70)

    try:
        hub.train_all(X_train, y_train, X_test, y_test, verbose=True)
    except Exception as e:
        print(f"[错误] 批量训练过程中出错: {e}\n")

    # ========== 步骤5: 批量评估 ==========
    print("[步骤5] 批量评估模型...")
    print("-" * 70)

    try:
        hub.evaluate_all(X_test, y_test, label_mapping=label_mapping, verbose=True)
    except Exception as e:
        print(f"[错误] 批量评估过程中出错: {e}\n")

    # ========== 步骤6: 生成报告 ==========
    print("[步骤6] 生成报告...")
    print("-" * 70)

    hub.print_comparison_table()
    hub.save_comparison_table()
    hub.plot_comparison_charts()
    hub.generate_all_reports()
    hub.generate_final_report()

    # ========== 步骤7: 最佳模型推荐 ==========
    print("[步骤7] 最佳模型推荐...")
    print("-" * 70)

    if len(hub.get_all_models()) > 0:
        _ = hub.recommend_best_model(criterion='f1_macro')

    print("\n" + "="*70)
    print("运行完成！")
    print("="*70)
    print(f"\n输出目录: {hub.output_dir}\n")


if __name__ == '__main__':
    main()
