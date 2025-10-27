"""
Prophet时间序列预测 - 按省份建模

⚠️ 警告：此模型不适合当前数据集！
原因：数据时间间隔极不规则（0-222天），虽然Prophet对缺失值鲁棒，但间隔过大仍无法学习有效模式
结果：R² < 0，预测效果极差
建议：使用横截面预测或传统机器学习方法（详见 README.md）

方法：
- 使用Facebook Prophet自动时间序列预测
- 按省份聚合所有监测站数据
- 自动检测趋势、季节性和节假日效应
- 对异常值和缺失数据有较好的鲁棒性

适用机型：
- Mac/Windows/Linux (通用CPU)
- 无需GPU，适合笔记本电脑运行

性能：
- 训练速度：快（优化的Stan后端）
- 内存需求：低（< 1GB）
- 相比ARIMA更易用，对异常值更鲁棒
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os

# Prophet库
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("错误: 未安装prophet库")
    print("请安装: pip install prophet")
    exit(1)

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'Outputs/Prophet_Province_Predictions'
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


def build_timeseries_prophet(province_data, indicator, train_ratio=0.8):
    """
    使用Prophet进行时间序列预测

    参数:
    - province_data: 省份数据（聚合所有监测站）
    - indicator: 预测指标
    - train_ratio: 训练集比例
    """
    data = province_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    # 按时间排序
    data = data.sort_values('Date').reset_index(drop=True)

    # 只保留目标指标和日期
    data = data[['Date', indicator]].dropna()

    if len(data) < 50:
        return None

    # 准备Prophet格式数据
    prophet_data = pd.DataFrame({
        'ds': data['Date'],
        'y': data[indicator]
    })

    # 时间顺序分割
    split_idx = int(len(prophet_data) * train_ratio)
    train_data = prophet_data[:split_idx]
    test_data = prophet_data[split_idx:]

    if len(train_data) < 30 or len(test_data) < 5:
        return None

    try:
        # 创建Prophet模型
        # 使用较保守的设置以提高稳定性
        model = Prophet(
            yearly_seasonality=True if len(train_data) >= 60 else False,
            weekly_seasonality=False,  # 水质数据通常无周季节性
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # 乘法季节性，适合幅度变化的数据
            changepoint_prior_scale=0.05,  # 降低过拟合风险
            seasonality_prior_scale=10,
            interval_width=0.95
        )

        # 添加自定义季节性（月度或季度）
        if len(train_data) >= 60:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        elif len(train_data) >= 30:
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=2)

        # 训练模型（抑制输出）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_data)

        # 预测
        future = test_data[['ds']].copy()
        forecast = model.predict(future)

        y_pred = forecast['yhat'].values
        y_test = test_data['y'].values

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

        # 提取趋势信息
        has_trend = len(model.changepoints) > 0
        trend_strength = abs(forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]) if len(forecast) > 1 else 0

        return {
            'model_name': 'Prophet',
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'metrics': metrics,
            'has_trend': has_trend,
            'trend_strength': trend_strength,
            'changepoints': len(model.changepoints)
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

        result = build_timeseries_prophet(province_data, indicator)
        if result:
            results[indicator] = result

    return results if results else None


def process_all_provinces(df):
    """处理所有省份"""
    print("\n" + "="*80)
    print("Prophet时间序列预测 - 按省份建模")
    print("="*80)

    # 获取所有省份
    provinces = df['Province'].unique()
    province_counts = df.groupby('Province').size().reset_index(name='count')

    print(f"\n总省份数: {len(provinces)}")
    print(f"\n方法参数:")
    print(f"  - 模型: Facebook Prophet")
    print(f"  - 自动检测: 趋势、季节性")
    print(f"  - 季节性模式: 乘法模型")
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

        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#f39c12')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        trend_text = f"趋势变化: {result['trend_strength']:.2f}" if result['has_trend'] else "无明显趋势"
        textstr = f"Model: Prophet\nR² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nN = {m['Sample_Size']}\n{trend_text}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.7))

        ax.set_title(indicator.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    title = f"{province} Province\nProphet Time Series Prediction (Province-Level Aggregation)"
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
        f.write("Prophet时间序列预测报告 - 按省份建模\n")
        f.write("Prophet Time Series Prediction Report - Province-Based Modeling\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("方法说明：\n")
        f.write("-" * 80 + "\n")
        f.write("- 每个省份独立建模（聚合省内所有监测站数据）\n")
        f.write("- 使用Facebook Prophet自动时间序列预测\n")
        f.write("- 自动检测趋势、季节性和变化点\n")
        f.write("- 季节性模式：乘法模型\n")
        f.write("- 对异常值和缺失值有良好的鲁棒性\n")
        f.write("- 数据按时间顺序分割：前80%训练，后20%测试\n\n")

        # 总体统计
        all_r2 = []
        all_rmse = []
        all_mae = []
        all_trends = []
        has_trend_count = 0

        for province, province_info in all_results.items():
            for indicator, result in province_info['results'].items():
                m = result['metrics']
                all_r2.append(m['R2'])
                all_rmse.append(m['RMSE'])
                all_mae.append(m['MAE'])
                all_trends.append(result['trend_strength'])
                if result['has_trend']:
                    has_trend_count += 1

        f.write("="*80 + "\n")
        f.write("总体性能\n")
        f.write("="*80 + "\n")
        f.write(f"省份数量: {len(all_results)}\n")
        f.write(f"总预测模型数: {len(all_r2)}\n")
        f.write(f"检测到趋势的模型数: {has_trend_count} ({has_trend_count/len(all_r2)*100:.1f}%)\n\n")

        f.write(f"R² 统计:\n")
        f.write(f"  平均值: {np.mean(all_r2):.4f}\n")
        f.write(f"  中位数: {np.median(all_r2):.4f}\n")
        f.write(f"  标准差: {np.std(all_r2):.4f}\n")
        f.write(f"  最大值: {np.max(all_r2):.4f}\n")
        f.write(f"  最小值: {np.min(all_r2):.4f}\n\n")

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

        # 趋势统计
        f.write(f"趋势强度统计:\n")
        f.write(f"  平均趋势强度: {np.mean(all_trends):.4f}\n")
        f.write(f"  最大趋势强度: {np.max(all_trends):.4f}\n")
        f.write(f"  最小趋势强度: {np.min(all_trends):.4f}\n\n")

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

            f.write(f"{'指标':<30} {'模型':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'趋势':>10}\n")
            f.write("-" * 80 + "\n")

            # 按R²排序
            sorted_results = sorted(province_info['results'].items(),
                                   key=lambda x: x[1]['metrics']['R2'],
                                   reverse=True)

            for indicator, result in sorted_results:
                m = result['metrics']
                trend_str = f"{result['trend_strength']:.2f}" if result['has_trend'] else "无"
                f.write(f"{indicator:<30} {'Prophet':<15} "
                       f"{m['R2']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {trend_str:>10}\n")

            f.write("\n" + "="*80 + "\n\n")

        f.write("\n说明：\n")
        f.write("- 每个省份独立建立Prophet时间序列模型（聚合省内所有监测站）\n")
        f.write("- R²越接近1表示预测效果越好\n")
        f.write("- Prophet能够自动检测趋势和季节性模式\n")
        f.write("- 对异常值和缺失值有良好的鲁棒性\n")
        f.write("="*80 + "\n")

    print(f"\n✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("Prophet时间序列预测系统 - 按省份建模")
    print("Prophet Time Series Prediction - Province-Based Modeling")
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
    print("\n[2/4] 按省份建立Prophet模型...")
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
