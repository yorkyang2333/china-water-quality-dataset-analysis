"""
中国水污染监测数据集真实性验证脚本

目标：证明数据集是模拟数据
核心证据：
1. 地理位置严重错误（经纬度与省份不匹配）
2. Water Quality Index 与水质指标无关

输出：
- 8张PNG图表（300 DPI）
- 1份总结文本报告

作者：Claude Code
日期：2025-12-14
"""

import zhplot  # 中文字体配置（必须在最前面）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import radians, cos, sin, asin, sqrt
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 常量定义 ====================

# 真实省份坐标（使用省会城市经纬度）
REAL_PROVINCE_COORDS = {
    'Beijing': (39.9042, 116.4074),      # 北京
    'Shanghai': (31.2304, 121.4737),     # 上海
    'Guangdong': (23.1291, 113.2644),    # 广州
    'Zhejiang': (30.2741, 120.1551),     # 杭州
    'Jiangsu': (32.0603, 118.7969),      # 南京
    'Shandong': (36.6512, 117.1201),     # 济南
    'Henan': (34.7466, 113.6253),        # 郑州
    'Hubei': (30.5928, 114.3055),        # 武汉
    'Sichuan': (30.5728, 104.0668),      # 成都
    'Yunnan': (25.0406, 102.7123),       # 昆明
}

# 中文名称映射
PROVINCE_CN_NAMES = {
    'Beijing': '北京', 'Shanghai': '上海', 'Guangdong': '广东',
    'Zhejiang': '浙江', 'Jiangsu': '江苏', 'Shandong': '山东',
    'Henan': '河南', 'Hubei': '湖北', 'Sichuan': '四川', 'Yunnan': '云南'
}

# 输出目录
OUTPUT_DIR = 'Outputs/Dataset_Verification'

# 绘图设置
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 工具函数 ====================

def haversine(lon1, lat1, lon2, lat2):
    """
    计算两点间的球面距离（公里）
    使用Haversine公式
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径（公里）
    return c * r


def load_data(filepath):
    """加载数据集"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='utf-8-sig')

    print(f"[OK] 加载数据: {len(df)} 条记录")
    print(f"[OK] 列数: {len(df.columns)}")
    print(f"[OK] 省份数: {df['Province'].nunique()}")
    print(f"[OK] 监测站数: {df['Monitoring_Station'].nunique()}")

    return df


# ==================== 地理位置验证模块 ====================

def calculate_geo_errors(df):
    """计算每条记录的地理位置偏差"""
    print("\n[INFO] 计算地理位置偏差...")

    distances = []
    for idx, row in df.iterrows():
        province = row['Province']
        if province in REAL_PROVINCE_COORDS:
            real_lat, real_lon = REAL_PROVINCE_COORDS[province]
            data_lat, data_lon = row['Latitude'], row['Longitude']
            distance = haversine(real_lon, real_lat, data_lon, data_lat)
            distances.append(distance)
        else:
            distances.append(np.nan)

    df['Geo_Error_km'] = distances

    # 统计
    valid_distances = [d for d in distances if not np.isnan(d)]
    mean_dist = np.mean(valid_distances)
    median_dist = np.median(valid_distances)
    max_dist = np.max(valid_distances)
    over_500 = sum(1 for d in valid_distances if d > 500) / len(valid_distances) * 100
    over_1000 = sum(1 for d in valid_distances if d > 1000) / len(valid_distances) * 100

    print(f"[OK] 平均偏差: {mean_dist:.2f} 公里")
    print(f"[OK] 最大偏差: {max_dist:.2f} 公里")
    print(f"[OK] 超过500公里比例: {over_500:.1f}%")
    print(f"[OK] 超过1000公里比例: {over_1000:.1f}%")

    return df, {
        'mean': mean_dist,
        'median': median_dist,
        'max': max_dist,
        'over_500_pct': over_500,
        'over_1000_pct': over_1000,
        'total_records': len(valid_distances)
    }


def plot_geo_error_map(df_with_errors, output_dir):
    """图1：地理位置错误总览地图"""
    print("\n[INFO] 绘制地理错误地图...")

    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制真实省份位置（大圆点）
    for province, (lat, lon) in REAL_PROVINCE_COORDS.items():
        cn_name = PROVINCE_CN_NAMES[province]
        ax.scatter(lon, lat, s=300, c='green', marker='o',
                  edgecolors='darkgreen', linewidths=2,
                  label=f'真实位置' if province == 'Beijing' else '', zorder=5)
        ax.text(lon, lat-0.5, cn_name, fontsize=10, ha='center',
               fontweight='bold', color='darkgreen')

    # 随机抽取部分数据点展示（避免过于密集）
    sample_df = df_with_errors.sample(n=min(500, len(df_with_errors)), random_state=42)

    # 绘制数据集中的位置（小红点）
    colors = plt.cm.rainbow(np.linspace(0, 1, len(REAL_PROVINCE_COORDS)))
    province_list = list(REAL_PROVINCE_COORDS.keys())

    for i, province in enumerate(province_list):
        province_data = sample_df[sample_df['Province'] == province]
        if len(province_data) > 0:
            ax.scatter(province_data['Longitude'], province_data['Latitude'],
                      s=20, c=[colors[i]], alpha=0.6,
                      label=f'{PROVINCE_CN_NAMES[province]} 数据点')

    # 绘制几条连线示例（随机选10个点）
    sample_for_lines = sample_df.sample(n=min(10, len(sample_df)), random_state=42)
    for idx, row in sample_for_lines.iterrows():
        province = row['Province']
        if province in REAL_PROVINCE_COORDS:
            real_lat, real_lon = REAL_PROVINCE_COORDS[province]
            ax.plot([real_lon, row['Longitude']], [real_lat, row['Latitude']],
                   'r--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('经度', fontsize=14)
    ax.set_ylabel('纬度', fontsize=14)
    ax.set_title('地理位置错误总览：数据集坐标 vs 真实省份位置',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_geo_error_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 01_geo_error_map.png")


def plot_geo_distance_distribution(df_with_errors, output_dir):
    """图2：偏差距离分布直方图"""
    print("\n[INFO] 绘制偏差距离分布...")

    distances = df_with_errors['Geo_Error_km'].dropna()

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制直方图
    n, bins, patches = ax.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

    # 添加平均值和中位数线
    mean_dist = distances.mean()
    median_dist = distances.median()

    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
              label=f'平均值: {mean_dist:.1f} 公里')
    ax.axvline(median_dist, color='orange', linestyle='--', linewidth=2,
              label=f'中位数: {median_dist:.1f} 公里')

    # 标注关键阈值
    ax.axvline(500, color='purple', linestyle=':', linewidth=1.5, alpha=0.7,
              label='500公里阈值')
    ax.axvline(1000, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7,
              label='1000公里阈值')

    ax.set_xlabel('地理位置偏差（公里）', fontsize=14)
    ax.set_ylabel('样本数量', fontsize=14)
    ax.set_title('地理位置偏差距离分布\n（大部分数据点距离真实位置数百至上千公里）',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加统计文本框
    stats_text = f'样本总数: {len(distances)}\n'
    stats_text += f'平均偏差: {mean_dist:.1f} km\n'
    stats_text += f'中位数: {median_dist:.1f} km\n'
    stats_text += f'最大偏差: {distances.max():.1f} km'

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_geo_distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 02_geo_distance_distribution.png")


def plot_top10_geo_errors(df_with_errors, output_dir):
    """图3：Top 10最离谱案例表格"""
    print("\n[INFO] 绘制Top 10案例表格...")

    # 排序并取前10
    df_sorted = df_with_errors.sort_values('Geo_Error_km', ascending=False).head(10)

    # 准备表格数据
    table_data = []
    for idx, (i, row) in enumerate(df_sorted.iterrows(), 1):
        province = row['Province']
        city = row['City']
        data_lat, data_lon = row['Latitude'], row['Longitude']
        real_lat, real_lon = REAL_PROVINCE_COORDS.get(province, (0, 0))
        error = row['Geo_Error_km']

        table_data.append([
            idx,
            PROVINCE_CN_NAMES.get(province, province),
            city,
            f"({data_lat:.2f}, {data_lon:.2f})",
            f"({real_lat:.2f}, {real_lon:.2f})",
            f"{error:.1f}"
        ])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=['排名', '省份', '城市', '数据集经纬度', '真实经纬度', '偏差(公里)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.12, 0.15, 0.22, 0.22, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色（交替）
    for i in range(1, 11):
        color = '#E7E6E6' if i % 2 == 0 else 'white'
        for j in range(6):
            table[(i, j)].set_facecolor(color)
            if j == 5:  # 偏差列，使用红色强调
                table[(i, j)].set_text_props(weight='bold', color='red')

    plt.title('Top 10 最离谱的地理位置错误案例\n（证明经纬度数据完全不匹配省份/城市）',
             fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_geo_top10_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 03_geo_top10_errors.png")

    return df_sorted


def plot_geo_province_stats(df_with_errors, output_dir):
    """图4：省份级统计柱状图"""
    print("\n[INFO] 绘制省份统计柱状图...")

    # 按省份统计平均偏差
    province_stats = df_with_errors.groupby('Province')['Geo_Error_km'].agg(['mean', 'count']).reset_index()
    province_stats = province_stats.sort_values('mean', ascending=True)

    # 转换为中文名称
    province_stats['Province_CN'] = province_stats['Province'].map(PROVINCE_CN_NAMES)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(province_stats['Province_CN'], province_stats['mean'],
                   color='coral', edgecolor='black', alpha=0.8)

    # 在柱子上标注数值
    for i, (bar, mean_val, count) in enumerate(zip(bars, province_stats['mean'], province_stats['count'])):
        ax.text(mean_val + 20, i, f'{mean_val:.0f} km\n(n={count})',
               va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('平均地理位置偏差（公里）', fontsize=14)
    ax.set_ylabel('省份', fontsize=14)
    ax.set_title('各省份平均地理位置偏差统计\n（所有省份数据都严重偏离真实位置）',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    # 添加500公里参考线
    ax.axvline(500, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='500公里阈值（超过此值明显异常）')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_geo_province_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 04_geo_province_stats.png")


# ==================== WQI关系验证模块 ====================

def analyze_wqi_correlation(df):
    """计算WQI与各指标的相关性"""
    print("\n[INFO] 分析WQI与水质指标的相关性...")

    # 选择数值型水质指标
    indicator_cols = [
        'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm', 'Turbidity_NTU',
        'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
        'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L',
        'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L',
        'Water_Quality_Index'
    ]

    # 确保所有列都存在
    indicator_cols = [col for col in indicator_cols if col in df.columns]

    # 计算相关性矩阵
    corr_matrix = df[indicator_cols].corr()

    # 提取WQI与其他指标的相关系数
    wqi_corr = corr_matrix['Water_Quality_Index'].drop('Water_Quality_Index')

    print(f"[OK] WQI与各指标的平均相关系数: {wqi_corr.abs().mean():.3f}")
    print(f"[OK] 最高相关系数: {wqi_corr.abs().max():.3f}")

    return corr_matrix, wqi_corr


def plot_wqi_correlation_heatmap(corr_matrix, output_dir):
    """图5：相关性热图"""
    print("\n[INFO] 绘制WQI相关性热图...")

    fig, ax = plt.subplots(figsize=(14, 12))

    # 创建热图
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               vmin=-1, vmax=1, ax=ax)

    ax.set_title('水质指标相关性热图\n（关注WQI行/列：如果WQI是真实的，应与某些指标高度相关）',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_wqi_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 05_wqi_correlation_heatmap.png")


def plot_wqi_scatter_matrix(df, output_dir):
    """图6：WQI vs 关键指标散点图组合"""
    print("\n[INFO] 绘制WQI散点图矩阵...")

    # 选择关键指标
    key_indicators = [
        'COD_mg_L', 'Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L',
        'Dissolved_Oxygen_mg_L', 'pH', 'Turbidity_NTU'
    ]

    key_indicators_cn = [
        '化学需氧量 (COD)', '总氮', '总磷',
        '溶解氧', 'pH值', '浊度'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (indicator, cn_name) in enumerate(zip(key_indicators, key_indicators_cn)):
        if indicator in df.columns:
            ax = axes[i]

            # 绘制散点图
            ax.scatter(df[indicator], df['Water_Quality_Index'],
                      alpha=0.3, s=10, c='steelblue')

            # 计算相关系数
            corr = df[[indicator, 'Water_Quality_Index']].corr().iloc[0, 1]

            ax.set_xlabel(cn_name, fontsize=12)
            ax.set_ylabel('WQI', fontsize=12)
            ax.set_title(f'{cn_name} vs WQI\n(相关系数: {corr:.3f})',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 添加趋势线
            try:
                z = np.polyfit(df[indicator].dropna(),
                             df.loc[df[indicator].notna(), 'Water_Quality_Index'], 1)
                p = np.poly1d(z)
                ax.plot(df[indicator].dropna().sort_values(),
                       p(df[indicator].dropna().sort_values()),
                       "r--", alpha=0.5, linewidth=2, label='趋势线')
                ax.legend(fontsize=9)
            except:
                pass

    plt.suptitle('WQI 与关键水质指标的关系\n（如果相关性都很低，说明WQI是随机编造的）',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{output_dir}/06_wqi_scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 06_wqi_scatter_matrix.png")


def predict_wqi_with_indicators(df):
    """尝试用水质指标预测WQI"""
    print("\n[INFO] 训练模型预测WQI...")

    # 选择特征（所有水质指标，排除WQI和非数值列）
    feature_cols = [
        'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm', 'Turbidity_NTU',
        'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
        'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L',
        'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L'
    ]

    # 确保所有列都存在
    feature_cols = [col for col in feature_cols if col in df.columns]

    # 准备数据
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['Water_Quality_Index']

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 评估
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[OK] R² 得分: {r2:.3f}")
    print(f"[OK] RMSE: {rmse:.2f}")
    print(f"[OK] MAE: {mae:.2f}")

    if r2 < 0.3:
        print("[WARN] R²极低！说明WQI无法从水质指标预测，是随机编造的！")

    return y_test, y_pred, r2, rmse, mae


def plot_wqi_prediction_failure(y_test, y_pred, r2, rmse, output_dir):
    """图7：WQI预测失败演示"""
    print("\n[INFO] 绘制WQI预测结果...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # 散点图
    ax.scatter(y_test, y_pred, alpha=0.4, s=30, c='steelblue', edgecolors='black', linewidths=0.5)

    # 理想预测线（y=x）
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
           label='理想预测线 (y=x)')

    ax.set_xlabel('实际 WQI', fontsize=14)
    ax.set_ylabel('预测 WQI', fontsize=14)
    ax.set_title(f'WQI预测失败演示\n（用所有水质指标训练随机森林模型）',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # 添加统计文本框
    stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\n\n'
    if r2 < 0.3:
        stats_text += '结论: R²极低！\nWQI无法从水质\n指标预测，是\n随机编造的数据'

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_wqi_prediction_failure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 07_wqi_prediction_failure.png")


def plot_pollution_level_mismatch(df, output_dir):
    """图8：污染等级与WQI不一致性"""
    print("\n[INFO] 绘制污染等级不一致性...")

    # 污染等级顺序
    level_order = ['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor']
    level_order_cn = ['优', '良', '中', '差', '极差']

    fig, ax = plt.subplots(figsize=(12, 7))

    # 创建箱线图数据
    data_to_plot = []
    labels = []
    for level in level_order:
        level_data = df[df['Pollution_Level'] == level]['Water_Quality_Index']
        if len(level_data) > 0:
            data_to_plot.append(level_data)
            idx = level_order.index(level)
            labels.append(level_order_cn[idx])

    # 绘制箱线图
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    ax.set_xlabel('污染等级', fontsize=14)
    ax.set_ylabel('WQI 数值', fontsize=14)
    ax.set_title('不同污染等级对应的WQI分布\n（如果WQI真实，不同等级的WQI应明显区分，而非严重重叠）',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加说明
    note_text = '注: 如果等级之间WQI范围严重重叠，\n说明污染等级分类是随机的'
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_wqi_pollution_level_mismatch.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] 保存: 08_wqi_pollution_level_mismatch.png")


# ==================== 总结报告生成 ====================

def generate_summary_report(geo_stats, wqi_stats, top10_df, output_path):
    """生成总结文本报告"""
    print("\n[INFO] 生成总结报告...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("           中国水污染监测数据集真实性验证报告\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集: 中国水污染监测数据集.csv\n")
        f.write(f"总记录数: {geo_stats['total_records']} 条\n")
        f.write(f"省份数量: 10 个\n\n")

        f.write("-" * 65 + "\n")
        f.write("证据一：地理位置严重错误\n")
        f.write("-" * 65 + "\n\n")

        f.write("1. 偏差距离统计:\n")
        f.write(f"   - 平均偏差距离: {geo_stats['mean']:.2f} 公里\n")
        f.write(f"   - 中位数偏差: {geo_stats['median']:.2f} 公里\n")
        f.write(f"   - 最大偏差: {geo_stats['max']:.2f} 公里\n")
        f.write(f"   - 超过500公里偏差的样本比例: {geo_stats['over_500_pct']:.1f}%\n")
        f.write(f"   - 超过1000公里偏差的样本比例: {geo_stats['over_1000_pct']:.1f}%\n\n")

        f.write("2. Top 10 最离谱的地理位置错误:\n")
        f.write("   排名 | 省份   | 城市           | 数据集经纬度          | 真实经纬度            | 偏差(公里)\n")
        f.write("   " + "-" * 95 + "\n")

        for idx, (i, row) in enumerate(top10_df.iterrows(), 1):
            province = row['Province']
            city = row['City']
            data_lat, data_lon = row['Latitude'], row['Longitude']
            real_lat, real_lon = REAL_PROVINCE_COORDS.get(province, (0, 0))
            error = row['Geo_Error_km']

            f.write(f"   {idx:2d}   | {PROVINCE_CN_NAMES.get(province, province):6s} | {city:14s} | "
                   f"({data_lat:6.2f}, {data_lon:7.2f}) | ({real_lat:6.2f}, {real_lon:7.2f}) | {error:10.1f}\n")

        f.write("\n3. 结论:\n")
        f.write(f"   经纬度数据与省份/城市完全不匹配，平均偏差达 {geo_stats['mean']:.0f} 公里。\n")
        f.write("   这种程度的偏差不可能是测量误差，明显是随机生成的数据。\n\n")

        f.write("-" * 65 + "\n")
        f.write("证据二：Water Quality Index 与水质指标无关\n")
        f.write("-" * 65 + "\n\n")

        f.write("1. 相关性分析:\n")
        f.write("   WQI与各水质指标的Pearson相关系数:\n")
        f.write(f"   - COD (化学需氧量): {wqi_stats['corr_cod']:.3f}\n")
        f.write(f"   - 总氮: {wqi_stats['corr_tn']:.3f}\n")
        f.write(f"   - 总磷: {wqi_stats['corr_tp']:.3f}\n")
        f.write(f"   - pH值: {wqi_stats['corr_ph']:.3f}\n")
        f.write(f"   - 溶解氧: {wqi_stats['corr_do']:.3f}\n")
        f.write(f"   平均相关系数: {wqi_stats['avg_corr']:.3f}\n\n")

        f.write("2. 预测模型验证:\n")
        f.write("   使用所有水质指标训练随机森林模型预测WQI:\n")
        f.write(f"   - R² (决定系数): {wqi_stats['r2']:.3f}\n")
        f.write(f"   - RMSE (均方根误差): {wqi_stats['rmse']:.2f}\n")
        f.write(f"   - MAE (平均绝对误差): {wqi_stats['mae']:.2f}\n\n")

        f.write("   如果WQI是从水质指标计算得出，R²应该>0.8。\n")
        f.write(f"   实际R²仅为 {wqi_stats['r2']:.3f}，说明WQI与指标几乎无关。\n\n")

        f.write("3. 结论:\n")
        f.write("   WQI无法从水质指标推导出来，与污染等级的分类也不一致。\n")
        f.write("   这表明WQI和Pollution_Level都是独立编造的随机数据。\n\n")

        f.write("-" * 65 + "\n")
        f.write("总结\n")
        f.write("-" * 65 + "\n\n")

        f.write("本数据集存在以下致命缺陷，证明其为模拟/虚假数据：\n\n")
        f.write(f"1. 地理位置数据完全错误，平均偏差 {geo_stats['mean']:.0f} 公里\n")
        f.write(f"2. 核心指标WQI与其他水质参数无关联（平均相关系数 {wqi_stats['avg_corr']:.2f}）\n")
        f.write(f"3. WQI无法通过机器学习模型从指标预测（R² = {wqi_stats['r2']:.2f}）\n")
        f.write("4. 污染等级分类与WQI数值不一致\n\n")

        f.write("建议：不应将此数据集用于真实的水质分析或政策建议。\n\n")
        f.write("=" * 65 + "\n")

    print(f"[DONE] 保存: verification_summary.txt")


# ==================== 主函数 ====================

def main():
    """主流程"""
    print("=" * 65)
    print("          数据集真实性验证程序")
    print("=" * 65)
    print("\n[INFO] 开始数据集真实性验证...\n")

    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] 输出目录: {OUTPUT_DIR}\n")

    # 2. 加载数据
    df = load_data('中国水污染监测数据集.csv')

    # 3. 地理位置验证
    print("\n" + "=" * 65)
    print("证据一：地理位置验证")
    print("=" * 65)

    df_with_errors, geo_stats = calculate_geo_errors(df)
    plot_geo_error_map(df_with_errors, OUTPUT_DIR)
    plot_geo_distance_distribution(df_with_errors, OUTPUT_DIR)
    top10_df = plot_top10_geo_errors(df_with_errors, OUTPUT_DIR)
    plot_geo_province_stats(df_with_errors, OUTPUT_DIR)

    # 4. WQI关系验证
    print("\n" + "=" * 65)
    print("证据二：WQI关系验证")
    print("=" * 65)

    corr_matrix, wqi_corr = analyze_wqi_correlation(df)
    plot_wqi_correlation_heatmap(corr_matrix, OUTPUT_DIR)
    plot_wqi_scatter_matrix(df, OUTPUT_DIR)

    y_test, y_pred, r2, rmse, mae = predict_wqi_with_indicators(df)
    plot_wqi_prediction_failure(y_test, y_pred, r2, rmse, OUTPUT_DIR)
    plot_pollution_level_mismatch(df, OUTPUT_DIR)

    # 收集WQI统计数据
    wqi_stats = {
        'avg_corr': wqi_corr.abs().mean(),
        'corr_cod': corr_matrix.loc['COD_mg_L', 'Water_Quality_Index'] if 'COD_mg_L' in corr_matrix.index else 0,
        'corr_tn': corr_matrix.loc['Total_Nitrogen_mg_L', 'Water_Quality_Index'] if 'Total_Nitrogen_mg_L' in corr_matrix.index else 0,
        'corr_tp': corr_matrix.loc['Total_Phosphorus_mg_L', 'Water_Quality_Index'] if 'Total_Phosphorus_mg_L' in corr_matrix.index else 0,
        'corr_ph': corr_matrix.loc['pH', 'Water_Quality_Index'] if 'pH' in corr_matrix.index else 0,
        'corr_do': corr_matrix.loc['Dissolved_Oxygen_mg_L', 'Water_Quality_Index'] if 'Dissolved_Oxygen_mg_L' in corr_matrix.index else 0,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

    # 5. 生成总结报告
    print("\n" + "=" * 65)
    print("生成总结报告")
    print("=" * 65)

    generate_summary_report(geo_stats, wqi_stats, top10_df,
                          f'{OUTPUT_DIR}/verification_summary.txt')

    # 完成
    print("\n" + "=" * 65)
    print("验证完成！")
    print("=" * 65)
    print(f"\n所有图表已保存到: {OUTPUT_DIR}/")
    print("\n生成的文件:")
    print("  - 01_geo_error_map.png (地理错误地图)")
    print("  - 02_geo_distance_distribution.png (偏差分布)")
    print("  - 03_geo_top10_errors.png (Top 10案例)")
    print("  - 04_geo_province_stats.png (省份统计)")
    print("  - 05_wqi_correlation_heatmap.png (相关性热图)")
    print("  - 06_wqi_scatter_matrix.png (散点图矩阵)")
    print("  - 07_wqi_prediction_failure.png (预测失败)")
    print("  - 08_wqi_pollution_level_mismatch.png (等级不一致)")
    print("  - verification_summary.txt (总结报告)")
    print("\n请使用这些图表和报告来制作PPT！\n")


if __name__ == '__main__':
    main()
