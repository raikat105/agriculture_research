import pandas as pd
import numpy as np
from datetime import timedelta

# === Step 1: Read datasets ===
print("Loading datasets...")
climate = pd.read_csv("C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\merged\\BANKURA.csv")
yield_data = pd.read_csv("C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\bankura_wheat_yield_2001_2024.csv")

# Parse date
climate['system:time_start'] = pd.to_datetime(climate['system:time_start'])

# === Step 2: Define Analysis Logic ===

def get_season_data(year, climate_df):
    # Season: Nov 15 (Year-1) to Apr 15 (Year)
    start_date = pd.Timestamp(year=year-1, month=11, day=15)
    end_date = pd.Timestamp(year=year, month=4, day=15)
    
    mask = (climate_df['system:time_start'] >= start_date) & (climate_df['system:time_start'] <= end_date)
    season_df = climate_df.loc[mask].copy()
    
    if season_df.empty:
        return None
        
    # Calculate days since start of season
    season_df['days_since_start'] = (season_df['system:time_start'] - start_date).dt.days
    
    # Define Week and Fortnight
    season_df['week'] = (season_df['days_since_start'] // 7) + 1
    season_df['fortnight'] = (season_df['days_since_start'] // 14) + 1
    
    return season_df

# Initialize list to store processed rows
processed_data = []

print("Processing years...")
# Iterate through each year in yield data
for year in yield_data['Year'].unique():
    season_df = get_season_data(year, climate)
    
    if season_df is not None:
        # Calculate Weekly Means
        weekly_means = season_df.groupby('week')[['precipitation', 'LST']].mean()
        # Flatten and rename
        weekly_features = {}
        for week, row in weekly_means.iterrows():
            weekly_features[f'week_{week}_precip'] = row['precipitation']
            weekly_features[f'week_{week}_LST'] = row['LST']
            
        # Calculate Fortnightly Means
        fortnightly_means = season_df.groupby('fortnight')[['precipitation', 'LST']].mean()
        # Flatten and rename
        fortnightly_features = {}
        for fn, row in fortnightly_means.iterrows():
            fortnightly_features[f'fortnight_{fn}_precip'] = row['precipitation']
            fortnightly_features[f'fortnight_{fn}_LST'] = row['LST']
            
        # Combine all features
        year_features = {'year': year}
        year_features.update(weekly_features)
        year_features.update(fortnightly_features)
        
        processed_data.append(year_features)

# Create DataFrame from processed data
climate_features_df = pd.DataFrame(processed_data)

# === Step 3: Merge with Yield Data ===
print("Merging data...")
# Ensure column name matches for merge
yield_data['year'] = yield_data['Year']

merged = pd.merge(yield_data, climate_features_df, on='year', how='left')

# === Step 4: Save Result ===
output_file = "bankura_wheat_detailed.csv"
merged.to_csv(output_file, index=False)

print(f"Detailed analysis saved as '{output_file}'")
print(f"Columns: {len(merged.columns)}")
print(merged.head())
