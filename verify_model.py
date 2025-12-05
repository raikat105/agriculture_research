import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def verify_modeling():
    print("Starting verification...")
    try:
        data = pd.read_csv("bankura_wheat_detailed.csv")
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: 'bankura_wheat_detailed.csv' not found.")
        return

    # The name of your yield column
    YIELD_COLUMN_NAME = 'Wheat_Yield_t_ha'

    # Identify feature columns automatically
    # We exclude metadata columns and the target variable
    exclude_cols = ['Year', 'District', 'Wheat_Yield_t_ha', 'year']
    features = [col for col in data.columns if col not in exclude_cols]

    print(f"Identified {len(features)} features for modeling.")

    # Drop any rows with missing data to ensure the model runs smoothly
    data.dropna(subset=features + [YIELD_COLUMN_NAME], inplace=True)

    X = data[features]
    y = data[YIELD_COLUMN_NAME]

    # Split the data into 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Data prepared for modeling:")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")

    # Initialize the model with 100 decision trees
    # random_state=42 ensures that the results are reproducible
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train (fit) the model on the training data
    rf_model.fit(X_train, y_train)

    print("Random Forest model has been successfully trained!")

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Calculate performance metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    print("Model Performance Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"R-squared (R²): {r2:.3f}")

    # Get feature importances
    importances = rf_model.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display top 10 features
    print("\nTop 10 Feature Importance Rankings:")
    print(feature_importance_df.head(10))

if __name__ == "__main__":
    verify_modeling()
