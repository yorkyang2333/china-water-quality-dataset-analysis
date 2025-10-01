import pandas as pd
import numpy as np

def determine_valid_ranges(input_filepath):
    """
    This function analyzes the preprocessed data and determines
    the valid ranges for each water quality indicator based on
    the actual min and max values in the dataset.
    """
    # Load the preprocessed dataset
    try:
        df = pd.read_csv(input_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_filepath, encoding='utf-8-sig')

    print(f"Total number of rows in dataset: {len(df)}")
    print(f"\n{'='*80}")
    print("Water Quality Indicator Valid Ranges Based on Preprocessed Data")
    print(f"{'='*80}\n")

    # Define the water quality indicator columns
    indicator_columns = [
        'Water_Temperature_C',
        'pH',
        'Dissolved_Oxygen_mg_L',
        'Conductivity_uS_cm',
        'Turbidity_NTU',
        'Nitrate_mg_L',
        'Nitrite_mg_L',
        'Ammonia_N_mg_L',
        'Total_Phosphorus_mg_L',
        'Total_Nitrogen_mg_L',
        'COD_mg_L',
        'BOD_mg_L',
        'Heavy_Metals_Pb_ug_L',
        'Heavy_Metals_Cd_ug_L',
        'Heavy_Metals_Hg_ug_L',
        'Coliform_Count_CFU_100mL'
    ]

    # Dictionary to store the valid ranges
    valid_ranges = {}

    # Calculate min and max for each indicator
    for col in indicator_columns:
        if col in df.columns:
            # Convert to numeric to ensure proper calculation
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate statistics
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()

            # Store in dictionary
            valid_ranges[col] = (min_val, max_val)

            # Print the results
            print(f"{col}:")
            print(f"  Min:    {min_val:.4f}")
            print(f"  Max:    {max_val:.4f}")
            print(f"  Mean:   {mean_val:.4f}")
            print(f"  Median: {median_val:.4f}")
            print(f"  Std:    {std_val:.4f}")
            print(f"  Range:  ({min_val:.2f}, {max_val:.2f})")
            print()
        else:
            print(f"{col}: Column not found in dataset")
            print()

    # Print the dictionary format for easy copying
    print(f"\n{'='*80}")
    print("Valid Ranges Dictionary (Python format):")
    print(f"{'='*80}\n")
    print("valid_ranges = {")
    for col, (min_val, max_val) in valid_ranges.items():
        print(f"    '{col}': ({min_val}, {max_val}),")
    print("}")

    # Save results to a text file
    output_file = 'EDA/valid_ranges_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Water Quality Indicator Valid Ranges\n")
        f.write("="*80 + "\n")
        f.write(f"Based on: {input_filepath}\n")
        f.write(f"Total rows: {len(df)}\n\n")

        for col in indicator_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()

                f.write(f"{col}:\n")
                f.write(f"  Min:    {min_val:.4f}\n")
                f.write(f"  Max:    {max_val:.4f}\n")
                f.write(f"  Mean:   {mean_val:.4f}\n")
                f.write(f"  Median: {median_val:.4f}\n")
                f.write(f"  Std:    {std_val:.4f}\n")
                f.write(f"  Range:  ({min_val:.2f}, {max_val:.2f})\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Valid Ranges Dictionary (Python format):\n")
        f.write("="*80 + "\n\n")
        f.write("valid_ranges = {\n")
        for col, (min_val, max_val) in valid_ranges.items():
            f.write(f"    '{col}': ({min_val}, {max_val}),\n")
        f.write("}\n")

    print(f"\nAnalysis results saved to '{output_file}'")

    return valid_ranges


if __name__ == '__main__':
    INPUT_FILE = 'preprocessed_data.csv'
    ranges = determine_valid_ranges(INPUT_FILE)