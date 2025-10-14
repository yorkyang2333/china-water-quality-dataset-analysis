import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import folium

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = 'Outputs/EDA'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(filepath):
    """Load preprocessed data"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(lambda x: 'Spring' if x in [3,4,5]
                                      else 'Summer' if x in [6,7,8]
                                      else 'Fall' if x in [9,10,11]
                                      else 'Winter')

    return df


def basic_statistics(df):
    """1. Basic data statistics"""
    print("="*80)
    print("1. DATA OVERVIEW")
    print("="*80)
    print(f"\nTotal Records: {len(df)}")
    print(f"Provinces: {df['Province'].nunique()}")
    print(f"Cities: {df['City'].nunique()}")
    print(f"Monitoring Stations: {df['Monitoring_Station'].nunique()}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nPollution Level Distribution:")
    print(df['Pollution_Level'].value_counts())

    # Create statistical charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pollution level pie chart
    pollution_counts = df['Pollution_Level'].value_counts()
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    axes[0].pie(pollution_counts.values, labels=pollution_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title('Distribution of Pollution Levels', fontsize=14, fontweight='bold')

    # Province data distribution
    province_counts = df['Province'].value_counts().head(10)
    axes[1].barh(province_counts.index, province_counts.values, color='steelblue')
    axes[1].set_xlabel('Number of Records')
    axes[1].set_title('Top 10 Provinces by Record Count', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_basic_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/01_basic_statistics.png")


def analyze_water_quality_indicators(df):
    """2. Water quality indicators distribution analysis"""
    print("\n" + "="*80)
    print("2. WATER QUALITY INDICATORS DISTRIBUTION")
    print("="*80)

    indicators = [
        'Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Conductivity_uS_cm',
        'Turbidity_NTU', 'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
        'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L', 'COD_mg_L', 'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L',
        'Coliform_Count_CFU_100mL'
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()

    for idx, indicator in enumerate(indicators):
        if indicator in df.columns:
            axes[idx].hist(df[indicator].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(indicator.replace('_', ' '), fontsize=9)
            axes[idx].set_ylabel('Frequency', fontsize=9)
            axes[idx].set_title(f'Distribution of {indicator.replace("_", " ")}', fontsize=10, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_indicators_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/02_indicators_distribution.png")


def analyze_pollution_by_region(df):
    """3. Regional pollution analysis"""
    print("\n" + "="*80)
    print("3. REGIONAL POLLUTION ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 3.1 Average water quality index by province
    province_wqi = df.groupby('Province')['Water_Quality_Index'].mean().sort_values(ascending=False).head(15)
    axes[0, 0].barh(province_wqi.index, province_wqi.values, color='coral')
    axes[0, 0].set_xlabel('Average Water Quality Index', fontsize=11)
    axes[0, 0].set_title('Top 15 Provinces by Average Water Quality Index', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Moderate Threshold')
    axes[0, 0].legend()

    # 3.2 Pollution level distribution by city
    city_pollution = df.groupby(['City', 'Pollution_Level']).size().unstack(fill_value=0)
    top_cities = df['City'].value_counts().head(10).index
    city_pollution_top = city_pollution.loc[top_cities]
    city_pollution_top.plot(kind='bar', stacked=True, ax=axes[0, 1],
                            color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
    axes[0, 1].set_xlabel('City', fontsize=11)
    axes[0, 1].set_ylabel('Number of Records', fontsize=11)
    axes[0, 1].set_title('Pollution Level Distribution by City (Top 10)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(title='Pollution Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3.3 Province pollution level percentage
    province_pollution_pct = df.groupby('Province')['Pollution_Level'].apply(
        lambda x: (x == 'Very Poor').sum() / len(x) * 100
    ).sort_values(ascending=False).head(15)
    axes[1, 0].barh(province_pollution_pct.index, province_pollution_pct.values, color='crimson')
    axes[1, 0].set_xlabel('Percentage of Very Poor Quality (%)', fontsize=11)
    axes[1, 0].set_title('Top 15 Provinces by Very Poor Water Quality Rate', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()

    # 3.4 Average heavy metal concentration by city
    city_heavy_metals = df.groupby('City')[['Heavy_Metals_Pb_ug_L',
                                             'Heavy_Metals_Cd_ug_L',
                                             'Heavy_Metals_Hg_ug_L']].mean()
    top_10_cities_hm = city_heavy_metals.sum(axis=1).sort_values(ascending=False).head(10).index
    city_heavy_metals.loc[top_10_cities_hm].plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_xlabel('City', fontsize=11)
    axes[1, 1].set_ylabel('Average Concentration (ug/L)', fontsize=11)
    axes[1, 1].set_title('Top 10 Cities by Average Heavy Metal Concentration', fontsize=12, fontweight='bold')
    axes[1, 1].legend(title='Heavy Metal', labels=['Pb', 'Cd', 'Hg'])
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03_regional_analysis.png")


def analyze_temporal_trends(df):
    """4. Temporal trend analysis"""
    print("\n" + "="*80)
    print("4. TEMPORAL TREND ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 4.1 Monthly average water quality index trend
    monthly_wqi = df.groupby('Month')['Water_Quality_Index'].mean()
    axes[0, 0].plot(monthly_wqi.index, monthly_wqi.values, marker='o', linewidth=2,
                    markersize=8, color='darkblue')
    axes[0, 0].fill_between(monthly_wqi.index, monthly_wqi.values, alpha=0.3)
    axes[0, 0].set_xlabel('Month', fontsize=11)
    axes[0, 0].set_ylabel('Average Water Quality Index', fontsize=11)
    axes[0, 0].set_title('Monthly Average Water Quality Index', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(1, 13))

    # 4.2 Seasonal pollution level distribution
    season_pollution = df.groupby(['Season', 'Pollution_Level']).size().unstack(fill_value=0)
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_pollution = season_pollution.reindex(season_order)
    season_pollution.plot(kind='bar', ax=axes[0, 1],
                          color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
    axes[0, 1].set_xlabel('Season', fontsize=11)
    axes[0, 1].set_ylabel('Number of Records', fontsize=11)
    axes[0, 1].set_title('Seasonal Pollution Level Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend(title='Pollution Level')
    axes[0, 1].tick_params(axis='x', rotation=0)

    # 4.3 Monthly pollutant concentration trends
    pollutants = ['COD_mg_L', 'BOD_mg_L', 'Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L']
    for pollutant in pollutants:
        monthly_avg = df.groupby('Month')[pollutant].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o',
                       label=pollutant.replace('_', ' '), linewidth=2)
    axes[1, 0].set_xlabel('Month', fontsize=11)
    axes[1, 0].set_ylabel('Average Concentration (mg/L)', fontsize=11)
    axes[1, 0].set_title('Monthly Trends of Key Pollutants', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(1, 13))

    # 4.4 Seasonal temperature vs dissolved oxygen
    season_temp_do = df.groupby('Season')[['Water_Temperature_C', 'Dissolved_Oxygen_mg_L']].mean()
    season_temp_do = season_temp_do.reindex(season_order)
    x = np.arange(len(season_order))
    width = 0.35
    axes[1, 1].bar(x - width/2, season_temp_do['Water_Temperature_C'], width,
                   label='Temperature (C)', color='orange', alpha=0.8)
    ax2 = axes[1, 1].twinx()
    ax2.bar(x + width/2, season_temp_do['Dissolved_Oxygen_mg_L'], width,
            label='Dissolved Oxygen (mg/L)', color='lightblue', alpha=0.8)
    axes[1, 1].set_xlabel('Season', fontsize=11)
    axes[1, 1].set_ylabel('Temperature (C)', fontsize=11, color='orange')
    ax2.set_ylabel('Dissolved Oxygen (mg/L)', fontsize=11, color='blue')
    axes[1, 1].set_title('Seasonal Temperature vs Dissolved Oxygen', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(season_order)
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/04_temporal_analysis.png")


def analyze_pollution_causes(df):
    """5. Pollution cause correlation analysis"""
    print("\n" + "="*80)
    print("5. POLLUTION CAUSE CORRELATION ANALYSIS")
    print("="*80)

    # Select key pollution indicators
    key_indicators = [
        'pH', 'Dissolved_Oxygen_mg_L', 'Turbidity_NTU',
        'Nitrate_mg_L', 'Nitrite_mg_L', 'Ammonia_N_mg_L',
        'Total_Phosphorus_mg_L', 'Total_Nitrogen_mg_L',
        'COD_mg_L', 'BOD_mg_L', 'Heavy_Metals_Pb_ug_L',
        'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L',
        'Coliform_Count_CFU_100mL', 'Water_Quality_Index'
    ]

    # Calculate correlation matrix
    corr_matrix = df[key_indicators].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 8})
    ax.set_title('Correlation Matrix of Water Quality Indicators', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/05_correlation_matrix.png")

    # Find indicators most correlated with water quality index
    wqi_corr = corr_matrix['Water_Quality_Index'].drop('Water_Quality_Index').abs().sort_values(ascending=False)
    print(f"\nTop 5 Indicators Correlated with Water Quality Index:")
    for i, (indicator, corr) in enumerate(wqi_corr.head().items(), 1):
        print(f"{i}. {indicator}: {corr:.3f}")


def analyze_pollutant_impact(df):
    """6. Impact of pollutants on water quality index"""
    print("\n" + "="*80)
    print("6. POLLUTANT IMPACT ON WATER QUALITY")
    print("="*80)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.ravel()

    pollutants = [
        ('COD_mg_L', 'COD (mg/L)'),
        ('BOD_mg_L', 'BOD (mg/L)'),
        ('Total_Nitrogen_mg_L', 'Total Nitrogen (mg/L)'),
        ('Total_Phosphorus_mg_L', 'Total Phosphorus (mg/L)'),
        ('Ammonia_N_mg_L', 'Ammonia-N (mg/L)'),
        ('Dissolved_Oxygen_mg_L', 'Dissolved Oxygen (mg/L)'),
        ('pH', 'pH'),
        ('Turbidity_NTU', 'Turbidity (NTU)'),
        ('Coliform_Count_CFU_100mL', 'Coliform Count (CFU/100mL)')
    ]

    for idx, (pollutant, label) in enumerate(pollutants):
        if pollutant in df.columns:
            # Scatter plot with trend line
            x = df[pollutant].dropna()
            y = df.loc[x.index, 'Water_Quality_Index']

            axes[idx].scatter(x, y, alpha=0.5, s=20, color='steelblue')

            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            axes[idx].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

            axes[idx].set_xlabel(label, fontsize=10)
            axes[idx].set_ylabel('Water Quality Index', fontsize=10)
            axes[idx].set_title(f'Impact of {label} on WQI', fontsize=11, fontweight='bold')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_pollutant_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/06_pollutant_impact.png")


def analyze_heavy_metal_pollution(df):
    """7. Heavy metal pollution analysis"""
    print("\n" + "="*80)
    print("7. HEAVY METAL POLLUTION ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 7.1 Heavy metal concentration boxplot
    heavy_metals = ['Heavy_Metals_Pb_ug_L', 'Heavy_Metals_Cd_ug_L', 'Heavy_Metals_Hg_ug_L']
    heavy_metals_data = df[heavy_metals].values.flatten()
    heavy_metals_labels = np.repeat(['Pb', 'Cd', 'Hg'], len(df))
    hm_df = pd.DataFrame({'Metal': heavy_metals_labels, 'Concentration': heavy_metals_data})

    sns.boxplot(data=hm_df, x='Metal', y='Concentration', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_ylabel('Concentration (ug/L)', fontsize=11)
    axes[0, 0].set_title('Heavy Metal Concentration Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 7.2 Heavy metal pollution vs pollution level
    pollution_levels = ['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor']
    hm_by_pollution = df.groupby('Pollution_Level')[heavy_metals].mean()
    hm_by_pollution = hm_by_pollution.reindex(pollution_levels)
    hm_by_pollution.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#f39c12', '#3498db'])
    axes[0, 1].set_xlabel('Pollution Level', fontsize=11)
    axes[0, 1].set_ylabel('Average Concentration (ug/L)', fontsize=11)
    axes[0, 1].set_title('Heavy Metals by Pollution Level', fontsize=12, fontweight='bold')
    axes[0, 1].legend(labels=['Pb', 'Cd', 'Hg'])
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 7.3 Relationship between heavy metals
    axes[1, 0].scatter(df['Heavy_Metals_Pb_ug_L'], df['Heavy_Metals_Cd_ug_L'],
                      alpha=0.5, s=30, c=df['Water_Quality_Index'], cmap='RdYlGn_r')
    axes[1, 0].set_xlabel('Pb Concentration (ug/L)', fontsize=11)
    axes[1, 0].set_ylabel('Cd Concentration (ug/L)', fontsize=11)
    axes[1, 0].set_title('Pb vs Cd Concentration', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 7.4 Heavy metal exceedance rate (standards: Pb<10, Cd<5, Hg<1 ug/L)
    standards = {'Heavy_Metals_Pb_ug_L': 10, 'Heavy_Metals_Cd_ug_L': 5, 'Heavy_Metals_Hg_ug_L': 1}
    exceed_counts = []
    for metal, standard in standards.items():
        exceed_count = (df[metal] > standard).sum()
        exceed_pct = exceed_count / len(df) * 100
        exceed_counts.append(exceed_pct)

    metals_short = ['Pb', 'Cd', 'Hg']
    bars = axes[1, 1].bar(metals_short, exceed_counts, color=['#e74c3c', '#f39c12', '#9b59b6'])
    axes[1, 1].set_ylabel('Percentage Exceeding Standard (%)', fontsize=11)
    axes[1, 1].set_title('Heavy Metal Exceedance Rate', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, count in zip(bars, exceed_counts):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_heavy_metal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/07_heavy_metal_analysis.png")


def analyze_nutrient_pollution(df):
    """8. Nutrient pollution analysis (eutrophication)"""
    print("\n" + "="*80)
    print("8. NUTRIENT POLLUTION ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 8.1 Nitrogen-phosphorus ratio analysis
    df['N_P_Ratio'] = df['Total_Nitrogen_mg_L'] / (df['Total_Phosphorus_mg_L'] + 0.001)

    axes[0, 0].scatter(df['Total_Phosphorus_mg_L'], df['Total_Nitrogen_mg_L'],
                      c=df['Water_Quality_Index'], cmap='RdYlGn_r', alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Total Phosphorus (mg/L)', fontsize=11)
    axes[0, 0].set_ylabel('Total Nitrogen (mg/L)', fontsize=11)
    axes[0, 0].set_title('Nitrogen vs Phosphorus Concentration', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar.set_label('Water Quality Index', fontsize=10)

    # 8.2 Nutrients vs season
    nutrients = ['Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L', 'Nitrate_mg_L', 'Ammonia_N_mg_L']
    season_nutrients = df.groupby('Season')[nutrients].mean()
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_nutrients = season_nutrients.reindex(season_order)

    season_nutrients.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_xlabel('Season', fontsize=11)
    axes[0, 1].set_ylabel('Average Concentration (mg/L)', fontsize=11)
    axes[0, 1].set_title('Seasonal Nutrient Variations', fontsize=12, fontweight='bold')
    axes[0, 1].legend(labels=['Total N', 'Total P', 'Nitrate', 'Ammonia-N'], fontsize=9)
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 8.3 Eutrophication risk assessment (based on nitrogen and phosphorus)
    df['Eutrophication_Risk'] = 'Low'
    df.loc[(df['Total_Nitrogen_mg_L'] > 2.0) | (df['Total_Phosphorus_mg_L'] > 0.1), 'Eutrophication_Risk'] = 'Medium'
    df.loc[(df['Total_Nitrogen_mg_L'] > 3.0) | (df['Total_Phosphorus_mg_L'] > 0.15), 'Eutrophication_Risk'] = 'High'

    risk_counts = df['Eutrophication_Risk'].value_counts()
    colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
    risk_order = ['Low', 'Medium', 'High']
    risk_counts = risk_counts.reindex(risk_order)

    axes[1, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=colors_risk, startangle=90)
    axes[1, 0].set_title('Eutrophication Risk Distribution', fontsize=12, fontweight='bold')

    # 8.4 Average nutrient concentration by province
    province_nutrients = df.groupby('Province')[['Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L']].mean()
    top_provinces = province_nutrients.sum(axis=1).sort_values(ascending=False).head(10)
    province_nutrients.loc[top_provinces.index].plot(kind='barh', ax=axes[1, 1],
                                                      color=['#3498db', '#e74c3c'])
    axes[1, 1].set_xlabel('Average Concentration (mg/L)', fontsize=11)
    axes[1, 1].set_title('Top 10 Provinces by Nutrient Concentration', fontsize=12, fontweight='bold')
    axes[1, 1].legend(labels=['Total N', 'Total P'])
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_nutrient_pollution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/08_nutrient_pollution.png")


def analyze_organic_pollution(df):
    """9. Organic pollution analysis"""
    print("\n" + "="*80)
    print("9. ORGANIC POLLUTION ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 9.1 COD vs BOD relationship
    axes[0, 0].scatter(df['BOD_mg_L'], df['COD_mg_L'],
                      c=df['Water_Quality_Index'], cmap='RdYlGn_r', alpha=0.6, s=30)
    axes[0, 0].set_xlabel('BOD (mg/L)', fontsize=11)
    axes[0, 0].set_ylabel('COD (mg/L)', fontsize=11)
    axes[0, 0].set_title('BOD vs COD Relationship', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Add diagonal reference line (theoretically COD > BOD)
    max_val = max(df['BOD_mg_L'].max(), df['COD_mg_L'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='COD = BOD')
    axes[0, 0].legend()

    # 9.2 COD/BOD ratio distribution
    df['COD_BOD_Ratio'] = df['COD_mg_L'] / (df['BOD_mg_L'] + 0.001)
    axes[0, 1].hist(df['COD_BOD_Ratio'].clip(upper=10), bins=50,
                    color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('COD/BOD Ratio', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('COD/BOD Ratio Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(x=2.5, color='red', linestyle='--', alpha=0.7,
                      label='Typical ratio for domestic wastewater')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 9.3 Organic pollution vs dissolved oxygen
    axes[1, 0].scatter(df['COD_mg_L'], df['Dissolved_Oxygen_mg_L'],
                      alpha=0.5, s=30, color='steelblue')
    axes[1, 0].set_xlabel('COD (mg/L)', fontsize=11)
    axes[1, 0].set_ylabel('Dissolved Oxygen (mg/L)', fontsize=11)
    axes[1, 0].set_title('Organic Pollution vs Dissolved Oxygen', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Add trend line
    x = df['COD_mg_L'].dropna()
    y = df.loc[x.index, 'Dissolved_Oxygen_mg_L']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[1, 0].plot(x_line, p(x_line), "r--", linewidth=2)

    # 9.4 Organic pollution by city
    city_organic = df.groupby('City')[['COD_mg_L', 'BOD_mg_L']].mean()
    top_cities_organic = city_organic.sum(axis=1).sort_values(ascending=False).head(10)
    city_organic.loc[top_cities_organic.index].plot(kind='barh', ax=axes[1, 1],
                                                     color=['#e74c3c', '#f39c12'])
    axes[1, 1].set_xlabel('Average Concentration (mg/L)', fontsize=11)
    axes[1, 1].set_title('Top 10 Cities by Organic Pollution', fontsize=12, fontweight='bold')
    axes[1, 1].legend(labels=['COD', 'BOD'])
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_organic_pollution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/09_organic_pollution.png")


def analyze_physical_parameters(df):
    """10. Physical parameters analysis"""
    print("\n" + "="*80)
    print("10. PHYSICAL PARAMETERS ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 10.1 pH distribution and relationship with water quality index
    axes[0, 0].scatter(df['pH'], df['Water_Quality_Index'], alpha=0.5, s=30, color='mediumseagreen')
    axes[0, 0].set_xlabel('pH', fontsize=11)
    axes[0, 0].set_ylabel('Water Quality Index', fontsize=11)
    axes[0, 0].set_title('pH vs Water Quality Index', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=7, color='blue', linestyle='--', alpha=0.5, label='Neutral pH')
    axes[0, 0].axvspan(6.5, 8.5, alpha=0.1, color='green', label='Optimal range')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 10.2 Turbidity distribution
    pollution_levels = ['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor']
    turbidity_by_pollution = [df[df['Pollution_Level'] == level]['Turbidity_NTU'].values
                              for level in pollution_levels]

    bp = axes[0, 1].boxplot(turbidity_by_pollution, labels=pollution_levels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']):
        patch.set_facecolor(color)
    axes[0, 1].set_xlabel('Pollution Level', fontsize=11)
    axes[0, 1].set_ylabel('Turbidity (NTU)', fontsize=11)
    axes[0, 1].set_title('Turbidity Distribution by Pollution Level', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 10.3 Dissolved oxygen sufficiency analysis
    df['DO_Status'] = 'Sufficient'
    df.loc[df['Dissolved_Oxygen_mg_L'] < 6, 'DO_Status'] = 'Low'
    df.loc[df['Dissolved_Oxygen_mg_L'] < 4, 'DO_Status'] = 'Critical'

    do_counts = df['DO_Status'].value_counts()
    do_order = ['Sufficient', 'Low', 'Critical']
    do_counts = do_counts.reindex(do_order, fill_value=0)
    colors_do = ['#2ecc71', '#f39c12', '#e74c3c']

    bars = axes[1, 0].bar(do_order, do_counts.values, color=colors_do)
    axes[1, 0].set_ylabel('Number of Records', fontsize=11)
    axes[1, 0].set_title('Dissolved Oxygen Status Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bar, count in zip(bars, do_counts.values):
        height = bar.get_height()
        pct = count / len(df) * 100
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    # 10.4 Conductivity vs pollution level
    conductivity_by_pollution = df.groupby('Pollution_Level')['Conductivity_uS_cm'].mean()
    conductivity_by_pollution = conductivity_by_pollution.reindex(pollution_levels)

    bars = axes[1, 1].bar(pollution_levels, conductivity_by_pollution.values,
                         color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
    axes[1, 1].set_xlabel('Pollution Level', fontsize=11)
    axes[1, 1].set_ylabel('Average Conductivity (uS/cm)', fontsize=11)
    axes[1, 1].set_title('Conductivity by Pollution Level', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_physical_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/10_physical_parameters.png")


def analyze_time_series_feasibility(df):
    """11. Time series prediction feasibility analysis"""
    print("\n" + "="*80)
    print("11. TIME SERIES PREDICTION FEASIBILITY ANALYSIS")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 11.1 数据点数分布（每个监测站）
    station_counts = df.groupby(['Province', 'City', 'Monitoring_Station']).size()

    axes[0, 0].hist(station_counts.values, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=30, color='red', linestyle='--', linewidth=2, label='LSTM Min (30)')
    axes[0, 0].axvline(x=50, color='orange', linestyle='--', linewidth=2, label='Recommended (50)')
    axes[0, 0].set_xlabel('Number of Data Points per Station', fontsize=11)
    axes[0, 0].set_ylabel('Number of Stations', fontsize=11)
    axes[0, 0].set_title('Distribution of Data Points per Monitoring Station', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 添加统计信息
    textstr = f'Mean: {station_counts.mean():.1f}\nMedian: {station_counts.median():.1f}\nMin: {station_counts.min()}\nMax: {station_counts.max()}'
    axes[0, 0].text(0.7, 0.95, textstr, transform=axes[0, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 11.2 数据时间跨度分析
    date_ranges = df.groupby(['Province', 'City', 'Monitoring_Station'])['Date'].agg(['min', 'max'])
    date_ranges['days'] = (date_ranges['max'] - date_ranges['min']).dt.days

    axes[0, 1].hist(date_ranges['days'].values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Time Span (Days)', fontsize=11)
    axes[0, 1].set_ylabel('Number of Stations', fontsize=11)
    axes[0, 1].set_title('Time Span Distribution per Station', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    textstr = f'Mean: {date_ranges["days"].mean():.1f} days\nMedian: {date_ranges["days"].median():.1f} days'
    axes[0, 1].text(0.7, 0.95, textstr, transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 11.3 数据密度分析（数据点/天数）
    station_density = station_counts / (date_ranges['days'] + 1)

    axes[0, 2].hist(station_density.values, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(x=1, color='red', linestyle='--', linewidth=2, label='1 point/day (ideal)')
    axes[0, 2].set_xlabel('Data Density (points/day)', fontsize=11)
    axes[0, 2].set_ylabel('Number of Stations', fontsize=11)
    axes[0, 2].set_title('Data Density per Station', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(axis='y', alpha=0.3)

    # 11.4 满足不同模型要求的监测站数量
    model_requirements = {
        'LSTM (≥30)': (station_counts >= 30).sum(),
        'ARIMA (≥30)': (station_counts >= 30).sum(),
        'SARIMA (≥40)': (station_counts >= 40).sum(),
        'Prophet (≥30)': (station_counts >= 30).sum(),
        'AR (≥25)': (station_counts >= 25).sum(),
        'Total Stations': len(station_counts)
    }

    models = list(model_requirements.keys())[:-1]
    counts = [model_requirements[m] for m in models]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#27ae60']

    bars = axes[1, 0].bar(range(len(models)), counts, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=len(station_counts), color='red', linestyle='--',
                       linewidth=2, label=f'Total: {len(station_counts)}')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Number of Qualified Stations', fontsize=11)
    axes[1, 0].set_title('Stations Meeting Time Series Model Requirements', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 添加百分比标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / len(station_counts) * 100
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    # 11.5 示例监测站的时间序列模式
    # 选择数据点最多的监测站
    top_station = station_counts.idxmax()
    station_data = df[
        (df['Province'] == top_station[0]) &
        (df['City'] == top_station[1]) &
        (df['Monitoring_Station'] == top_station[2])
    ].sort_values('Date')

    if len(station_data) > 0:
        axes[1, 1].plot(station_data['Date'], station_data['Water_Quality_Index'],
                       marker='o', linestyle='-', linewidth=1, markersize=4, color='darkblue')
        axes[1, 1].set_xlabel('Date', fontsize=11)
        axes[1, 1].set_ylabel('Water Quality Index', fontsize=11)
        axes[1, 1].set_title(f'Time Series Example: {top_station[2]}\n({len(station_data)} points)',
                            fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

    # 11.6 结论面板
    axes[1, 2].axis('off')

    # 计算关键统计
    avg_points = station_counts.mean()
    median_points = station_counts.median()
    pct_sufficient = (station_counts >= 30).sum() / len(station_counts) * 100
    avg_density = station_density.mean()

    conclusion_text = "WHY THIS DATASET IS NOT SUITABLE\nFOR TIME SERIES PREDICTION\n\n"
    conclusion_text += "="*50 + "\n\n"
    conclusion_text += "CRITICAL ISSUES:\n\n"
    conclusion_text += f"1. INSUFFICIENT DATA POINTS\n"
    conclusion_text += f"   • Average: {avg_points:.1f} points/station\n"
    conclusion_text += f"   • Median: {median_points:.1f} points/station\n"
    conclusion_text += f"   • Requirement: ≥30-50 points\n"
    conclusion_text += f"   • Only {pct_sufficient:.1f}% stations qualified\n\n"

    conclusion_text += f"2. LOW DATA DENSITY\n"
    conclusion_text += f"   • Average: {avg_density:.2f} points/day\n"
    conclusion_text += f"   • Most stations have sparse data\n"
    conclusion_text += f"   • Irregular time intervals\n\n"

    conclusion_text += f"3. UNSUITABLE FOR:\n"
    conclusion_text += f"   ✗ LSTM (needs sequential patterns)\n"
    conclusion_text += f"   ✗ ARIMA (needs stationarity)\n"
    conclusion_text += f"   ✗ SARIMA (needs seasonality)\n"
    conclusion_text += f"   ✗ Prophet (needs trends)\n"
    conclusion_text += f"   ✗ AR (needs autocorrelation)\n\n"

    conclusion_text += f"RECOMMENDATION:\n"
    conclusion_text += f"   → Use cross-sectional models\n"
    conclusion_text += f"   → Use spatial features\n"
    conclusion_text += f"   → Use regression models\n"
    conclusion_text += f"   → Collect more temporal data\n"

    axes[1, 2].text(0.1, 0.95, conclusion_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_time_series_feasibility.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/11_time_series_feasibility.png")

    # 打印统计摘要
    print(f"\nTime Series Feasibility Summary:")
    print(f"  Total stations: {len(station_counts)}")
    print(f"  Avg points per station: {avg_points:.1f}")
    print(f"  Stations with ≥30 points: {(station_counts >= 30).sum()} ({pct_sufficient:.1f}%)")
    print(f"  Avg data density: {avg_density:.2f} points/day")
    print(f"\n  ⚠️  CONCLUSION: Dataset is NOT suitable for time series prediction!")


def generate_summary_report(df):
    """12. Generate summary report"""
    print("\n" + "="*80)
    print("12. GENERATING SUMMARY REPORT")
    print("="*80)

    report = []
    report.append("="*80)
    report.append("WATER POLLUTION EXPLORATORY DATA ANALYSIS - SUMMARY REPORT")
    report.append("="*80)
    report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: preprocessed_data.csv")
    report.append(f"Total Records: {len(df)}")
    report.append(f"\n{'-'*80}\n")

    # 1. Basic statistics
    report.append("1. BASIC STATISTICS")
    report.append(f"   - Provinces covered: {df['Province'].nunique()}")
    report.append(f"   - Cities monitored: {df['City'].nunique()}")
    report.append(f"   - Monitoring stations: {df['Monitoring_Station'].nunique()}")
    report.append(f"   - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # 2. Pollution level distribution
    report.append(f"\n2. POLLUTION LEVEL DISTRIBUTION")
    for level, count in df['Pollution_Level'].value_counts().items():
        pct = count / len(df) * 100
        report.append(f"   - {level}: {count} ({pct:.2f}%)")

    # 3. Key findings
    report.append(f"\n3. KEY FINDINGS")

    # Province with most severe pollution
    very_poor_by_province = df[df['Pollution_Level'] == 'Very Poor'].groupby('Province').size().sort_values(ascending=False)
    if len(very_poor_by_province) > 0:
        report.append(f"   - Province with most 'Very Poor' records: {very_poor_by_province.index[0]} ({very_poor_by_province.values[0]} records)")

    # Province with highest average water quality index
    avg_wqi_by_province = df.groupby('Province')['Water_Quality_Index'].mean().sort_values(ascending=False)
    report.append(f"   - Province with highest avg WQI: {avg_wqi_by_province.index[0]} ({avg_wqi_by_province.values[0]:.2f})")

    # Heavy metal exceedance
    standards = {'Heavy_Metals_Pb_ug_L': 10, 'Heavy_Metals_Cd_ug_L': 5, 'Heavy_Metals_Hg_ug_L': 1}
    for metal, standard in standards.items():
        exceed_count = (df[metal] > standard).sum()
        exceed_pct = exceed_count / len(df) * 100
        metal_name = metal.split('_')[2]
        report.append(f"   - {metal_name} exceeding standard: {exceed_count} records ({exceed_pct:.2f}%)")

    # Eutrophication risk
    high_risk = (df['Eutrophication_Risk'] == 'High').sum()
    high_risk_pct = high_risk / len(df) * 100
    report.append(f"   - High eutrophication risk: {high_risk} records ({high_risk_pct:.2f}%)")

    # Low dissolved oxygen
    low_do = (df['Dissolved_Oxygen_mg_L'] < 6).sum()
    low_do_pct = low_do / len(df) * 100
    report.append(f"   - Low dissolved oxygen (<6 mg/L): {low_do} records ({low_do_pct:.2f}%)")

    # 4. Pollution causes
    report.append(f"\n4. MAIN POLLUTION CAUSES")
    corr_with_wqi = df[['COD_mg_L', 'BOD_mg_L', 'Total_Nitrogen_mg_L', 'Total_Phosphorus_mg_L',
                        'Ammonia_N_mg_L', 'Heavy_Metals_Pb_ug_L', 'Turbidity_NTU',
                        'Water_Quality_Index']].corr()['Water_Quality_Index'].drop('Water_Quality_Index')
    top_corr = corr_with_wqi.abs().sort_values(ascending=False).head(5)

    report.append("   Top 5 indicators correlated with Water Quality Index:")
    for i, (indicator, corr) in enumerate(top_corr.items(), 1):
        report.append(f"   {i}. {indicator.replace('_', ' ')}: {abs(corr):.3f}")

    # 5. Seasonal characteristics
    report.append(f"\n5. SEASONAL CHARACTERISTICS")
    seasonal_avg_wqi = df.groupby('Season')['Water_Quality_Index'].mean()
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    for season in season_order:
        if season in seasonal_avg_wqi.index:
            report.append(f"   - {season}: Avg WQI = {seasonal_avg_wqi[season]:.2f}")

    # 6. Recommendations
    report.append(f"\n6. RECOMMENDATIONS")
    report.append("   - Focus on reducing organic pollution (COD/BOD) in highly polluted areas")
    report.append("   - Implement stricter controls on nitrogen and phosphorus discharge")
    report.append("   - Monitor heavy metal sources, especially in high-risk provinces")
    report.append("   - Improve wastewater treatment in cities with poor water quality")
    report.append("   - Establish seasonal monitoring strategies based on identified patterns")

    report.append(f"\n{'='*80}")
    report.append("END OF REPORT")
    report.append("="*80)

    # Save report
    report_text = '\n'.join(report)
    with open(f'{OUTPUT_DIR}/00_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved: {OUTPUT_DIR}/00_summary_report.txt")


def create_station_map(df):
    """12. Create a map of monitoring stations using Folium."""
    print("\n" + "="*80)
    print("12. CREATING MONITORING STATION MAP")
    print("="*80)

    # Define colors for each pollution level
    color_map = {
        'Excellent': 'green',
        'Good': 'blue',
        'Moderate': 'orange',
        'Poor': 'red',
        'Very Poor': 'darkred'
    }

    # Create a base map centered on China
    # Coordinates for the approximate center of China
    map_center = [35.8617, 104.1954]
    station_map = folium.Map(location=map_center, zoom_start=5)

    # Get unique stations to avoid plotting the same station multiple times
    # We take the first entry for each station to get its location and a representative pollution level
    unique_stations = df.drop_duplicates(subset=['Monitoring_Station'])

    # Add a marker for each station
    for _, row in unique_stations.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        
        # Ensure lat/lon are valid numbers
        if pd.isna(lat) or pd.isna(lon):
            continue

        station_name = row['Monitoring_Station']
        pollution_level = row['Pollution_Level']
        city = row['City']
        province = row['Province']

        # Create a popup with station information
        popup_html = f"""
        <b>Station:</b> {station_name}<br>
        <b>City:</b> {city}<br>
        <b>Province:</b> {province}<br>
        <b>Pollution Level:</b> {pollution_level}
        """
        popup = folium.Popup(popup_html, max_width=300)

        # Get color for the marker
        marker_color = color_map.get(pollution_level, 'gray') # Default to gray if level is unknown

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7,
            popup=popup
        ).add_to(station_map)

    # Save the map to an HTML file
    map_path = f'{OUTPUT_DIR}/12_station_distribution_map.html'
    station_map.save(map_path)
    print(f"\nSaved: {map_path}")


def main():
    """Main function"""
    print("\n" + "="*80)
    print(" " * 15 + "WATER POLLUTION EDA - COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")

    # Load data
    INPUT_FILE = 'preprocessed_data.csv'
    print(f"Loading data from: {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"Data loaded successfully: {len(df)} records\n")

    # Execute analysis
    basic_statistics(df)
    analyze_water_quality_indicators(df)
    analyze_pollution_by_region(df)
    analyze_temporal_trends(df)
    analyze_pollution_causes(df)
    analyze_pollutant_impact(df)
    analyze_heavy_metal_pollution(df)
    analyze_nutrient_pollution(df)
    analyze_organic_pollution(df)
    analyze_physical_parameters(df)
    analyze_time_series_feasibility(df)
    create_station_map(df) # Call the new map function
    generate_summary_report(df)

    print("\n" + "="*80)
    print(" " * 20 + "EDA COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll visualizations and reports saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for i in range(13):
        if i == 0:
            print(f"  - 00_summary_report.txt")
        else:
            files = {
                1: "01_basic_statistics.png",
                2: "02_indicators_distribution.png",
                3: "03_regional_analysis.png",
                4: "04_temporal_analysis.png",
                5: "05_correlation_matrix.png",
                6: "06_pollutant_impact.png",
                7: "07_heavy_metal_analysis.png",
                8: "08_nutrient_pollution.png",
                9: "09_organic_pollution.png",
                10: "10_physical_parameters.png",
                11: "11_time_series_feasibility.png",
                12: "12_station_distribution_map.html" # Add the new map file
            }
            if i in files:
                print(f"  - {files[i]}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()