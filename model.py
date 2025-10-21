import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Set a style for the plots
sns.set_style("whitegrid")

# === Step 1: Load your combined dataset ===
# Make sure this file is in the same directory as your script, or provide the full path.
try:
    data = pd.read_csv("bankura_wheat_combined.csv")
except FileNotFoundError:
    print("Error: 'bankura_wheat_combined.csv' not found.")
    print("Please ensure the CSV file created by your previous script is in the correct location.")
    # Exit the script if the file isn't found
    exit()

# --- Debugging Step: Print all column names to find the correct one ---
print("✅ Columns found in your CSV file are:")
print(data.columns.tolist())
print("-" * 30)
# ----------------------------------------------------------------------


# === Step 2: Prepare the data for the model ===

# 🚨 IMPORTANT: Copy the correct yield column name from the list printed above
# and paste it here. For example, if it's 'Yield', change the line to:
# YIELD_COLUMN_NAME = 'Yield'
YIELD_COLUMN_NAME = 'Wheat_Yield_t_ha'

# Check if the chosen column name exists before proceeding
if YIELD_COLUMN_NAME not in data.columns:
    print(f"❌ Error: The column '{YIELD_COLUMN_NAME}' was not found in the CSV.")
    print("Please update the YIELD_COLUMN_NAME variable with a name from the list above and run again.")
    exit()

# Drop rows with any missing values, which can happen from the merge
data.dropna(inplace=True)

# Define the features (X) and the target (y)
# Features are all the climate data we engineered.
features = [
    'LST_sowing', 'precipitation_sowing',
    'LST_growth', 'precipitation_growth',
    'LST_flowering', 'precipitation_flowering',
    'LST_harvest', 'precipitation_harvest'
]

X = data[features]
y = data[YIELD_COLUMN_NAME]

# Split data into training and testing sets
# We'll use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data prepared for the model:")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 30)


# === Step 3: Build and Train the Random Forest Model ===

# Initialize the model. n_estimators is the number of trees in the forest.
# random_state ensures we get the same results every time we run the code.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print("-" * 30)


# === Step 4: Evaluate the Model's Performance ===

# Make predictions on the unseen test data
y_pred = rf_model.predict(X_test)

# Calculate performance metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("📈 Model Performance Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("(R² is the proportion of the variance in yield that is predictable from the climate features. Closer to 1 is better.)")
print("-" * 30)


# === Step 5: Analyze Feature Importance ===
# This is the key step to answer your question!

print("🔍 Analyzing feature importance...")

# Get importance scores from the trained model
importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False) # Sort by most important

print("\nFeature Importance Rankings:")
print(feature_importance_df)

# Visualize the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')

plt.title('Which Climate Factors Drastically Affect Wheat Yield?', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Climate Factor & Crop Phase', fontsize=12)
plt.tight_layout() # Adjust layout to make sure everything fits
plt.show()