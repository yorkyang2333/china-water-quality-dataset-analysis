
import pandas as pd
import matplotlib.pyplot as plt

def check_data_quality(file_path):
    """
    Checks a CSV file for missing values and duplicate rows, then visualizes the results.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:        
        # --- Matplotlib configuration for Chinese characters ---
        plt.rcParams['font.sans-serif'] = ['SimSong']  # Specify a font that supports Chinese
        plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs

        df = pd.read_csv(file_path)
        
        # Exclude the 'remarks' column from the missing value count
        if 'Remarks' in df.columns:
            missing_values_count = df.drop(columns=['Remarks']).isnull().sum().sum()
        else:
            missing_values_count = df.isnull().sum().sum()
        
        duplicate_rows_count = df.duplicated().sum()
        
        # --- Visualization ---
        labels = ['缺失值总数', '重复行数']
        counts = [missing_values_count, duplicate_rows_count]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, counts, color=['#4c72b0', '#c44e52'])

        # Add titles and labels
        ax.set_ylabel('数量')
        ax.set_title('数据质量检查报告')

        # Add count labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

        plt.show()

    except FileNotFoundError:
        print(f"错误: 文件未找到 at path: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 使用你的CSV文件的实际路径
    csv_file = '中国水污染监测数据集.csv'
    check_data_quality(csv_file)
