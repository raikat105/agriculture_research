import pandas as pd

# Load Excel file (default: first sheet)
excel_file = 'C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\Rainfall_merge\\DARJEELING\\2000-2024.xlsx'
df = pd.read_excel(excel_file)

# Save as CSV
csv_file = 'C:\\Users\\RAIKAT\\OneDrive\\Documents\\Final Year Project\\Rainfall_merge\\DARJEELING\\2000-2024.csv'
df.to_csv(csv_file, index=False)

print(f"✅ Excel converted to CSV: {csv_file}")
