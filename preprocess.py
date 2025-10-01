import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def preprocess_data(input_filepath, output_filepath):
    """
    This function loads a dataset, removes outliers using the 3-sigma and IQR methods,
    and saves the cleaned data to a new CSV file.
    """
    # Load the dataset, trying different encodings
    try:
        df = pd.read_csv(input_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_filepath, encoding='gbk')

    print(f"Original number of rows: {len(df)}")

    """
    # Load API credentials from environment variables
    API_KEY = os.getenv("API_KEY")
    GEOCODING_URL = os.getenv("GEOCODING_URL", "https://restapi.amap.com/v3/geocode/geo")

    if not API_KEY:
        raise ValueError("API_KEY not found in environment variables. Please create a .env file with API_KEY.")

    # Get unique cities
    unique_cities = df['City'].unique()

    # Get city coordinates
    city_coords = {}
    for city in unique_cities:
        params = {
            "key": API_KEY,
            "address": city
        }
        try:
            response = requests.get(GEOCODING_URL, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            if data['status'] == '1' and data['geocodes']:
                location = data['geocodes'][0]['location']
                lon, lat = map(float, location.split(','))
                city_coords[city] = {"lat": lat, "lon": lon}
            else:
                print(f"Geocoding failed for {city}: {data.get('info', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching geocoding data for {city}: {e}")
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing geocoding data for {city}: {e}")

    # Filter rows based on coordinates
    def is_valid_coord(row):
        city = row['City']
        if city in city_coords:
            lat_diff = abs(row['Latitude'] - city_coords[city]['lat'])
            lon_diff = abs(row['Longitude'] - city_coords[city]['lon'])
            if lat_diff <= 10 and lon_diff <= 10:
                return True
            else:
                # print(f"Row for city {city} with coords ({row['Latitude']}, {row['Longitude']}) is being dropped. Correct coords are ({city_coords[city]['lat']}, {city_coords[city]['lon']})")
                return False
        else:
            # print(f"City {city} not found in geocoding results, row will be dropped.")
            return False

    original_rows = len(df)
    df = df[df.apply(is_valid_coord, axis=1)]
    print(f"Number of rows after coordinate validation: {len(df)} ({original_rows - len(df)} rows dropped)")
    """
    
    # Define the valid ranges for each water quality indicator
    # Ranges are based on natural/polluted freshwater standards and actual data distribution
    valid_ranges = {
        'Water_Temperature_C': (0, 40),           # Natural freshwater range
        'pH': (4.5, 10.0),                        # Natural water pH range (slightly acidic to alkaline)
        'Dissolved_Oxygen_mg_L': (0, 20),         # Saturation at ~14 mg/L, allowing for supersaturation
        'Conductivity_uS_cm': (50, 2000),         # Freshwater range (50-1500 μS/cm typical)
        'Turbidity_NTU': (0, 150),                # Clear to very turbid water
        'Nitrate_mg_L': (0, 100),                 # Polluted water can reach 50+ mg/L
        'Nitrite_mg_L': (0, 5),                   # Typically <1 mg/L, allowing for pollution events
        'Ammonia_N_mg_L': (0, 10),                # Polluted water range
        'Total_Phosphorus_mg_L': (0, 2),          # Eutrophic water range
        'Total_Nitrogen_mg_L': (0, 20),           # Eutrophic/polluted water range
        'COD_mg_L': (0, 300),                     # Polluted water range
        'BOD_mg_L': (0, 100),                     # Polluted water range
        'Heavy_Metals_Pb_ug_L': (0, 100),         # Allowing for contaminated sites
        'Heavy_Metals_Cd_ug_L': (0, 10),          # Allowing for contaminated sites
        'Heavy_Metals_Hg_ug_L': (0, 2),           # Strict limit due to toxicity
        'Coliform_Count_CFU_100mL': (0, 10000)    # From clean to heavily contaminated water
    }

    # Filter the dataframe based on the valid ranges
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            # Skip filtering for columns that are not in the dataframe
            if col not in df.columns:
                continue
            
            # Convert column to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN values that were introduced by coercion
            df.dropna(subset=[col], inplace=True)

            # Apply the filter
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]


    print(f"Number of rows after applying theoretical ranges: {len(df)}")

    # Identify numerical columns for outlier detection
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "Remarks" in numerical_cols:
        numerical_cols.remove("Remarks")
    
    if not numerical_cols:
        print("No numerical columns found for outlier detection.")
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        return

    # Create a boolean mask to identify rows to keep. Initialize to all True.
    rows_to_keep = pd.Series([True] * len(df), index=df.index)

    # Iterate over numerical columns to find outliers
    for col in numerical_cols:
        # 3-sigma method
        mean = df[col].mean()
        std = df[col].std()
        lower_bound_3sigma = mean - 3 * std
        upper_bound_3sigma = mean + 3 * std

        # IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR

        # Identify outliers from both methods for the current column
        # Outliers are values outside the union of the two valid ranges
        # Valid range for 3-sigma: [lower_bound_3sigma, upper_bound_3sigma]
        # Valid range for IQR: [lower_bound_iqr, upper_bound_iqr]
        # We take the wider (more inclusive) range of the two methods to avoid removing too much data.
        final_lower_bound = min(lower_bound_3sigma, lower_bound_iqr)
        final_upper_bound = max(upper_bound_3sigma, upper_bound_iqr)
        
        # Update the mask to mark rows with outliers in the current column as False
        rows_to_keep &= (df[col] >= final_lower_bound) & (df[col] <= final_upper_bound)

    # Filter the dataframe to keep only the desired rows
    df_cleaned = df[rows_to_keep]

    print(f"Number of rows after outlier removal: {len(df_cleaned)}")

    # Save the preprocessed data
    df_cleaned.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print(f"Preprocessing complete. Cleaned data saved to '{output_filepath}'")


if __name__ == '__main__':
    INPUT_FILE = '中国水污染监测数据集.csv'
    OUTPUT_FILE = 'preprocessed_data.csv'
    preprocess_data(INPUT_FILE, OUTPUT_FILE)
