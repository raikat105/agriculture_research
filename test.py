import pandas as pd
import numpy as np

def fill_missing_with_neighbors_mean(series):
    filled_series = []
    values = series.values
    for i in range(len(values)):
        if pd.isna(values[i]):
            neighbors = []
            if i - 2 >= 0 and not pd.isna(values[i - 2]):
                neighbors.append(values[i - 2])
            if i - 1 >= 0 and not pd.isna(values[i - 1]):
                neighbors.append(values[i - 1])
            if i + 1 < len(values) and not pd.isna(values[i + 1]):
                neighbors.append(values[i + 1])
            if i + 2 < len(values) and not pd.isna(values[i + 2]):
                neighbors.append(values[i + 2])
            filled_value = round(np.mean(neighbors), 2) if neighbors else 0.0
            filled_series.append(filled_value)
        else:
            filled_series.append(values[i])
    return pd.Series(filled_series, index=series.index)

def merge_and_fill_weather_data(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1['system:time_start'] = pd.to_datetime(df1['system:time_start'])
    df2['system:time_start'] = pd.to_datetime(df2['system:time_start'])

    merged_df = pd.merge(df1, df2, on='system:time_start', how='outer')
    merged_df = merged_df.sort_values(by='system:time_start')

    merged_df['precipitation'] = fill_missing_with_neighbors_mean(merged_df['precipitation'])
    merged_df['LST'] = fill_missing_with_neighbors_mean(merged_df['LST'])

    merged_df.to_csv(output_file, index=False)
    print(f"✅ Cleaned and merged CSV saved to: {output_file}")

# Example usage
merge_and_fill_weather_data(
    'C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\Rainfall_merge\\UTTAR DINAJPUR\\2000-2024.csv',
    'C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\Temperature_merge\\UTTAR DINAJPUR\\2000-2024.csv',
    'C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\merged\\UTTAR DINAJPUR.csv'
)
