import pandas as pd

# === Step 1: Read datasets ===
climate = pd.read_csv("C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\merged\\BANKURA.csv")
yield_data = pd.read_csv("C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\bankura_wheat_yield_2001_2024.csv")

# Parse date
climate['system:time_start'] = pd.to_datetime(climate['system:time_start'])
climate['year'] = climate['system:time_start'].dt.year
climate['month'] = climate['system:time_start'].dt.month
climate['day'] = climate['system:time_start'].dt.day

# === Step 2: Define crop phases ===
def get_phase(row):
    m, d = row['month'], row['day']
    if (m == 11 and d >= 15) or (m == 12 and d <= 15):
        return 'sowing'
    elif (m == 12 and d > 15) or (m in [1, 2] and d <= 15):
        return 'growth'
    elif (m == 2 and d > 15) or (m == 3 and d <= 15):
        return 'flowering'
    elif (m == 3 and d > 15) or (m == 4 and d <= 15):
        return 'harvest'
    else:
        return None

climate['phase'] = climate.apply(get_phase, axis=1)
climate = climate.dropna(subset=['phase'])

# === Step 3: Compute mean temperature & precipitation for each phase ===
phase_means = (
    climate.groupby(['year', 'phase'])[['precipitation', 'LST']]
    .mean()
    .reset_index()
    .pivot(index='year', columns='phase')
)

# Flatten columns
phase_means.columns = ['_'.join(col).strip() for col in phase_means.columns.values]
phase_means.reset_index(inplace=True)

# === Step 4: Use the correct year column to create a 'year' column ===
# The error traceback confirms that your yield data has a 'Year' column.
# Replace the placeholder with the correct column name.
yield_data['year'] = yield_data['Year']

# === Step 5: Merge with yield data ===
merged = pd.merge(yield_data, phase_means, on='year', how='left')

# === Step 6: Save the combined dataset ===
merged.to_csv("bankura_wheat_combined.csv", index=False)

print("✅ Combined dataset saved as 'bankura_wheat_combined.csv'")
print(merged.head())