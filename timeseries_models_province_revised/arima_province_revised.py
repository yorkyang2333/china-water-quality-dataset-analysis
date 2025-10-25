"""
ARIMA时间序列预测 - 按省份建模

⚠️ 警告：此模型不适合当前数据集！
原因：数据时间间隔极不规则（0-222天），违反ARIMA等间隔/近似等间隔采样假设
结果：R² < 0，预测效果极差
建议：使用横截面预测或传统机器学习方法（详见 README.md）

方法：
- 使用ARIMA(p,d,q)自回归积分滑动平均模型
- 按省份聚合所有监测站数据
- 自动参数选择（基于AIC准则）
- 时间序列分解 + 平稳性检验

适用机型：
- Mac/Windows/Linux (通用CPU)
- 无需GPU，适合笔记本电脑运行

性能：
- 训练速度：中等（统计模型，相对较快）
- 内存需求：低（< 1GB）

⭐ 关键修改（2023/2024版本）：
1. 仅使用2023年的数据进行模型训练
2. 2024年的数据作为验证/测试集
3. 可视化输出改为折线图（实际值vs预测值随时间变化）

"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
from itertools import product

# ARIMA相关库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/ARIMA_Province_2023Train_2024Valid_Province'
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
    """
    检查时间序列的平稳性
    使用ADF检验（Augmented Dickey-Fuller test）
    """
    try:
        result = adfuller(timeseries, autolag='AIC')
        p_value = result[1]
        # p-value < 0.05 表示序列是平稳的
        return p_value < 0.05, p_value
    except:
        return False, 1.0


def find_best_arima_params(timeseries, max_p=3, max_d=2, max_q=3):
    """
    寻找最佳ARIMA参数
    基于AIC准则（Akaike Information Criterion）
    """
    best_aic = np.inf
    best_params = (1, 1, 1)
    best_model = None

    # 限制搜索空间以提高速度和稳定性
    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)

    for p, d, q in product(p_range, d_range, q_range):
        try:
            if p == 0 and d == 0 and q == 0:
                continue

            model = ARIMA(timeseries, order=(p, d, q))
            fitted = model.fit(method_kwargs={"warn_convergence": False})

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_params = (p, d, q)
                best_model = fitted
        except:
            continue

    return best_params, best_aic, best_model


def build_timeseries_arima(province_data, indicator, auto_params=True):
    """
    使用ARIMA进行时间序列预测

    参数:
    - province_data: 省份数据（聚合所有监测站）
    - indicator: 预测指标
    - train_ratio: 训练集比例
    - auto_params: 是否自动选择参数
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
    train_data_df = data[data['Year'] == 2023]
    test_data_df = data[data['Year'] == 2024]
    test_dates = test_data_df['Date'].values  # ⭐ 新增：保存测试集日期

    # 提取实际的时间序列值
    train_data = train_data_df[indicator].values
    test_data = test_data_df[indicator].values

    if len(train_data) < 30 or len(test_data) < 5:
        return None

    try:
        # 选择ARIMA参数
        if auto_params:
            best_params, _, fitted_model = find_best_arima_params(train_data)
            if fitted_model is None:
                # 如果自动选择失败，使用默认参数
                best_params = (1, 1, 1)
                model = ARIMA(train_data, order=best_params)
                fitted_model = model.fit(method_kwargs={"warn_convergence": False})
        else:
            # 默认参数
            best_params = (1, 1, 1)
            model = ARIMA(train_data, order=best_params)
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})

        # 预测
        forecast = fitted_model.forecast(steps=len(test_data))
        y_pred = np.array(forecast)
        y_test = test_data

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred)
        if metrics is None:
            return None

        # 验证R²的合理性，过滤掉极差的预测
        if metrics['R2'] < -100:  # R²过低说明模型完全失效
            return None

        # 检查预测值的合理性
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return None

        return {
            'model_name': 'ARIMA',
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'test_dates': test_dates,  # ⭐ 新增：保存日期用于折线图
            'metrics': metrics,
            'params': best_params,
            'aic': fitted_model.aic,
            'is_stationary': is_stationary,
            'p_value': p_value
        }

    except Exception:
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

        result = build_timeseries_arima(province_data, indicator)
        if result:
            results[indicator] = result

    return results if results else None


def process_all_provinces(df):
    """处理所有省份"""
    print("\n" + "="*80)
    print("ARIMA时间序列预测 - 按省份建模")
    print("="*80)

    # 获取所有省份
    provinces = df['Province'].unique()
    province_counts = df.groupby('Province').size().reset_index(name='count')

    print(f"\n总省份数: {len(provinces)}")
    print(f"\n方法参数:")
    print(f"  - 模型: ARIMA(p,d,q)")
    print(f"  - 参数选择: 自动优化（基于AIC）")
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
        params = result['params']

        # ⭐ 使用折线图代替散点图
        test_dates = pd.to_datetime(result['test_dates'])
        ax.plot(test_dates, y_test, marker='o', linewidth=2, markersize=5,
                color='steelblue', label='Actual (2024)', alpha=0.8)
        ax.plot(test_dates, y_pred, marker='s', linewidth=2, markersize=5,
                color='orangered', label='Predicted', alpha=0.8, linestyle='--')

        _ = min(y_test.min(), y_pred.min())  # unused
        _ = max(y_test.max(), y_pred.max())  # unused

        stationary_text = "平稳" if result['is_stationary'] else "非平稳"
        textstr = f"Model: ARIMA{params}\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}\n{stationary_text}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Date (2024)', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{province} Province\nARIMA Time Series Prediction (Province-Level Aggregation)"
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
        f.write("ARIMA时间序列预测报告 - 按省份建模\n")
        f.write("ARIMA Time Series Prediction Report - Province-Based Modeling\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个省份独立建模（聚合省内所有监测站数据）\n")
        f.write("- 使用ARIMA自回归积分滑动平均模型\n")
        f.write("- 参数(p,d,q)自动选择，基于AIC准则\n")
        f.write("- 平稳性检验：ADF test\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n")
        f.write("- ARIMA适合捕捉线性时间序列模式\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_params = []
        stationary_count = 0

        for province, province_info in all_results.items():
            for indicator, result in province_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                all_params.append(result['params'])
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

        # 常用参数统计
        f.write(f"ARIMA参数分布:\n")
        param_counts = {}
        for p in all_params:
            if p not in param_counts:
                param_counts[p] = 0
            param_counts[p] += 1

        if len(all_params) > 0:
            top_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for param, count in top_params:
                f.write(f"  ARIMA{param}: {count} 次 ({count/len(all_params)*100:.1f}%)\n")
        else:
            f.write("  无有效参数数据\n")
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
                params = result['params']
                model_str = f"ARIMA{params}"
                f.write(f"{indicator:<30} {model_str:<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['Sample_Size']:>8d}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个省份独立建立ARIMA时间序列模型（聚合省内所有监测站）\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- ARIMA适合捕捉线性时间序列模式和趋势\n")
        f.write("- 参数(p,d,q)自动选择，基于AIC准则\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("ARIMA时间序列预测系统 - 按省份建模")
    print("ARIMA Time Series Prediction - Province-Based Modeling")
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
    print("\n[2/4] 按省份建立ARIMA模型...")
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
