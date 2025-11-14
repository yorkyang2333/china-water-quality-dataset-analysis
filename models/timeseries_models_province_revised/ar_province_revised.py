"""
AR自回归模型时间序列预测 - 按省份建模

⚠️ 警告：此模型不适合当前数据集！
原因：数据时间间隔极不规则（0-222天），违反AR等间隔采样假设
结果：R² < 0，预测效果极差
建议：使用横截面预测或传统机器学习方法（详见 README.md）

方法：
- 使用AR(p)自回归模型
- 按省份聚合所有监测站数据
- 自动选择最优滞后阶数p（基于AIC/BIC）
- 时间序列分解 + 平稳性检验

适用机型：
- Mac/Windows/Linux (通用CPU)
- 无需GPU，适合笔记本电脑运行

性能：
- 训练速度：快（比ARIMA更简单）
- 内存需求：低（< 1GB）
- 模型简单，不易过拟合

⭐ 关键修改（2023/2024版本）：
1. 仅使用2023年的数据进行模型训练
2. 2024年的数据作为验证/测试集
3. 可视化输出改为折线图（实际值vs预测值随时间变化）

"""

import zhplot  # 自动配置matplotlib中文字体
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os

# AR模型库
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

OUTPUT_DIR = 'Outputs/AR_Province_2023Train_2024Valid_Province'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return None

    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Sample_Size': len(y_true)
    }

def check_stationarity(timeseries):
    """检查时间序列的平稳性"""
    try:
        result = adfuller(timeseries, autolag='AIC')
        p_value = result[1]
        return p_value < 0.05, p_value
    except:
        return False, 1.0

def find_best_ar_lag(timeseries, maxlag=10):
    """
    寻找最优滞后阶数
    基于AIC准则
    """
    best_aic = np.inf
    best_lag = 1

    max_possible_lag = min(maxlag, len(timeseries) // 3)

    for lag in range(1, max_possible_lag + 1):
        try:
            model = AutoReg(timeseries, lags=lag, old_names=False)
            fitted = model.fit()

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_lag = lag
        except:
            continue

    return best_lag, best_aic

def build_timeseries_ar(province_data, indicator, train_ratio=0.8, auto_lag=True):
    """
    使用AR模型进行时间序列预测

    参数:
    - province_data: 省份数据（聚合所有监测站）
    - indicator: 预测指标
    - train_ratio: 训练集比例
    - auto_lag: 是否自动选择滞后阶数
    """
    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    # 按时间排序
    data = data.sort_values('Date').reset_index(drop=True)

    # 只保留目标指标和日期
    data = data[['Date', indicator]].dropna()

    if len(data) < 50:
        return None

    # 提取目标值
    values = data[indicator].values

    # 检查平稳性
    is_stationary, p_value = check_stationarity(values)

    # ⭐ 按年份分割（2023训练，2024测试）
    data['Year'] = data['Date'].dt.year
    train_data = data[data['Year'] == 2023]
    test_data = data[data['Year'] == 2024]

    if len(train_data) < 30 or len(test_data) < 5:
        return None

    try:
        # 选择滞后阶数
        if auto_lag:
            best_lag, aic = find_best_ar_lag(train_data[indicator].values, maxlag=min(15, len(train_data) // 4))
        else:
            best_lag = min(5, len(train_data) // 4)
            aic = None

        # 训练AR模型（使用指标列）
        model = AutoReg(train_data[indicator].values, lags=best_lag, old_names=False)
        fitted_model = model.fit()

        # 预测
        forecast = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        y_pred = np.array(forecast)
        y_test = test_data[indicator].values
        test_dates = test_data['Date'].values

        # 检查预测值的合理性
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return None

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred)
        if metrics is None:
            return None

        # 验证R²的合理性
        if metrics['R2'] < -100:
            return None

        return {
            'model_name': 'AR',
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'test_dates': test_dates,  # ⭐ 新增：保存日期用于折线图
            'metrics': metrics,
            'lag': best_lag,
            'aic': fitted_model.aic,
            'is_stationary': is_stationary,
            'p_value': p_value
        }

    except Exception as e:
        return None

def predict_province(df, province):
    """为单个省份预测所有指标（聚合所有监测站数据）"""
    province_data = df[df['Province'] == province].copy()

    if len(province_data) < 50:
        return None

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm', 'Turbidity_NTU', 'Nitrate_mg_L',
        'Nitrite_mg_L', 'Ammonia_N_mg_L', 'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L', 'Coliform_Count_CFU_100mL'
    ]

    results = {}

    for indicator in indicators:
        # 检查数据量
        valid_count = province_data[indicator].notna().sum()
        if valid_count < 40:
            continue

        result = build_timeseries_ar(province_data, indicator)
        if result:
            results[indicator] = result

    return results if results else None

def process_all_provinces(df):
    """处理所有省份"""
    print("\n" + "="*80)
    print("AR自回归模型时间序列预测 - 按省份建模")
    print("="*80)

    # 获取所有省份
    provinces = df['Province'].unique()
    province_counts = df.groupby('Province').size().reset_index(name='count')

    print(f"\n总省份数: {len(provinces)}")
    print(f"\n方法参数:")
    print(f"  - 模型: AR(p) 自回归模型")
    print(f"  - 滞后阶数: 自动选择（基于AIC）")
    print(f"  - 平稳性检验: ADF test")
    print(f"  - 数据分割: 80% 训练, 20% 测试")
    print(f"  - 建模方式: 按省份聚合所有监测站数据")

    all_results = {}
    successful_provinces = 0

    for idx, province in enumerate(sorted(provinces)):
        count = province_counts[province_counts['Province'] == province]['count'].values[0]

        print(f"\n[{idx+1}/{len(provinces)}] {province}")
        print(f"  总数据点数: {count}")

        results = predict_province(df, province)

        if results:
            all_results[province] = {
                'province': province,
                'sample_count': count,
                'results': results
            }
            successful_provinces += 1

            # 显示结果摘要
            r2_scores = [r['metrics']['R2'] for r in results.values()]
            print(f"  ✓ 成功预测 {len(results)} 个指标")
            print(f"  平均 R²: {np.mean(r2_scores):.3f} (范围: {np.min(r2_scores):.3f} - {np.max(r2_scores):.3f})")
        else:
            print(f"  ✗ 数据不足，跳过")

    print(f"\n成功建模的省份: {successful_provinces}/{len(provinces)}")

    return all_results

def visualize_province(province, province_info, output_dir):
    """可视化单个省份的预测结果"""
    results = province_info['results']

    if not results:
        return

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(20, 5*nrows))

    for idx, (indicator, result) in enumerate(results.items()):
        ax = plt.subplot(nrows, ncols, idx + 1)

        y_test = result['y_test']
        y_pred = result['y_pred']
        m = result['metrics']
        lag = result['lag']

        # ⭐ 使用折线图代替散点图
        test_dates = pd.to_datetime(result['test_dates'])
        ax.plot(test_dates, y_test, marker='o', linewidth=2, markersize=5,
                color='steelblue', label='Actual (2024)', alpha=0.8)
        ax.plot(test_dates, y_pred, marker='s', linewidth=2, markersize=5,
                color='orangered', label='Predicted', alpha=0.8, linestyle='--')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())

        stationary_text = "平稳" if result['is_stationary'] else "非平稳"
        textstr = f"Model: AR({lag})\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}\n{stationary_text}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='plum', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Date (2024)', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{province} Province\nAR Time Series Prediction (Province-Level Aggregation)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{output_dir}/{province}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(all_results, output_dir):
    """生成综合报告"""
    report_path = f'{output_dir}/report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AR自回归模型时间序列预测报告 - 按省份建模\n")
        f.write("AR Time Series Prediction Report - Province-Based Modeling\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个省份独立建模（聚合省内所有监测站数据）\n")
        f.write("- 使用AR(p)自回归模型\n")
        f.write("- 滞后阶数p自动选择，基于AIC准则\n")
        f.write("- 平稳性检验：ADF test\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n")
        f.write("- AR模型简单高效，适合短期预测\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_lags = []
        stationary_count = 0

        for province, province_info in all_results.items():
            for indicator, result in province_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                all_lags.append(result['lag'])
                if result['is_stationary']:
                    stationary_count += 1

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"省份数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n")
        if len(all_r2) > 0:
            f.write(f"平稳序列数: {stationary_count} ({stationary_count/len(all_r2)*100:.1f}%)\n\n")
        else:
            f.write(f"平稳序列数: {stationary_count} (无有效模型)\n\n")

        if len(all_r2) > 0:
            f.write(f"R² 统计:\n")
            f.write(f"  平均值: {np.mean(all_r2):.4f}\n")
            f.write(f"  中位数: {np.median(all_r2):.4f}\n")
            f.write(f"  标准差: {np.std(all_r2):.4f}\n")
            f.write(f"  最大值: {np.max(all_r2):.4f}\n")
            f.write(f"  最小值: {np.min(all_r2):.4f}\n\n")
        else:
            f.write("R² 统计: 无有效模型数据\n\n")

        # R²分布
        r2_ranges = {
            'R² > 0.7 (优秀)': sum(1 for r in all_r2 if r > 0.7),
            'R² 0.5-0.7 (良好)': sum(1 for r in all_r2 if 0.5 <= r <= 0.7),
            'R² 0.3-0.5 (中等)': sum(1 for r in all_r2 if 0.3 <= r < 0.5),
            'R² < 0.3 (较差)': sum(1 for r in all_r2 if r < 0.3)
        }

        f.write(f"R² 分布:\n")
        for range_name, count in r2_ranges.items():
            percentage = count / len(all_r2) * 100 if len(all_r2) > 0 else 0
            f.write(f"  {range_name}: {count} ({percentage:.1f}%)\n")
        f.write("\n")

        f.write(f"RMSE 平均值: {np.mean(all_rmse):.4f}\n")
        f.write(f"MAE 平均值: {np.mean(all_mae):.4f}\n\n")

        # 滞后阶数统计
        f.write(f"滞后阶数分布:\n")
        f.write(f"  平均滞后阶数: {np.mean(all_lags):.1f}\n")
        f.write(f"  最常用滞后阶数: {max(set(all_lags), key=all_lags.count)}\n")
        lag_counts = {}
        for lag in all_lags:
            if lag not in lag_counts:
                lag_counts[lag] = 0
            lag_counts[lag] += 1
        if len(all_lags) > 0:
            for lag in sorted(lag_counts.keys())[:5]:
                f.write(f"  AR({lag}): {lag_counts[lag]} 次 ({lag_counts[lag]/len(all_lags)*100:.1f}%)\n")
        else:
            f.write("  无有效滞后阶数数据\n")
        f.write("\n")

        f.write("\n" + "="*80 + "\n\n")

        # 详细的省份结果 - 按R²排序
        f.write("详细结果 - 按R²排序\n")
        f.write("="*80 + "\n\n")

        # 收集所有省份的平均R²
        province_avg_r2 = []
        for province, province_info in all_results.items():
            r2_list = [result['metrics']['R2'] for result in province_info['results'].values()]
            avg_r2 = np.mean(r2_list)
            province_avg_r2.append((province, province_info, avg_r2))

        # 按平均R²排序
        province_avg_r2.sort(key=lambda x: x[2], reverse=True)

        for province, province_info, avg_r2 in province_avg_r2:
            f.write(f"省份: {province}\n")
            f.write(f"样本数: {province_info['sample_count']} | 平均R²: {avg_r2:.4f}\n")
            f.write("-" * 80 + "\n")

            f.write(f"{'指标':<30} {'模型':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'样本':>8}\n")
            f.write("-" * 80 + "\n")

            # 按R²排序
            sorted_results = sorted(province_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                model_str = f"AR({result['lag']})"
                f.write(f"{indicator:<30} {model_str:<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个省份独立建立AR自回归模型（聚合省内所有监测站）\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- AR模型简单高效，适合短期线性预测\n")
        f.write("- 滞后阶数自动选择，基于AIC准则\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")

def main():
    """主函数"""
    print("="*80)
    print("AR自回归模型时间序列预测系统 - 按省份建模")
    print("AR Time Series Prediction - Province-Based Modeling")
    print("="*80)

    # 加载数据
    print("\n[1/4] 加载数据...")
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')
    except:
        df = pd.read_csv('preprocessed_data.csv', encoding='gbk')

    print(f"✓ 总记录数: {len(df)}")
    print(f"✓ 省份数: {df['Province'].nunique()}")
    print(f"✓ 监测站数: {df['Monitoring_Station'].nunique()}")

    # 处理所有省份
    print("\n[2/4] 按省份建立AR模型...")
    all_results = process_all_provinces(df)

    # 生成可视化
    print("\n[3/4] 生成可视化...")
    total = len(all_results)
    for idx, (province, province_info) in enumerate(all_results.items(), 1):
        print(f"  进度: {idx}/{total}")
        visualize_province(province, province_info, OUTPUT_DIR)

    # 生成报告
    print("\n[4/4] 生成报告...")
    generate_report(all_results, OUTPUT_DIR)

    # 总结
    print("\n" + "="*80)
    print("完成！")
    print("="*80)

    total_models = sum(len(p['results']) for p in all_results.values())
    print(f"\n✓ 成功建模的省份: {len(all_results)}")
    print(f"✓ 总预测模型数: {total_models}")
    print(f"✓ 结果保存在: {OUTPUT_DIR}/")
    print(f"  - 报告: report.txt")
    print(f"  - 图表: {len(all_results)} 个省份的可视化\n")

if __name__ == "__main__":
    main()
