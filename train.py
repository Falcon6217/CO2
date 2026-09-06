# train.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# 1. FILE PATHS
# ==========================================================

DATA_PATH = "sustainability_data.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "co2_emission_model.pkl")


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ==========================================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "Energy_Consumption",
    "Renewable_Percentage",
    "GDP",
    "CO2_Emissions"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ==========================================================
# 4. HANDLE MISSING VALUES
# ==========================================================

print("\nChecking missing values:")

print(data[required_columns].isnull().sum())

data = data.dropna(subset=required_columns)

print("\nDataset shape after removing missing values:")
print(data.shape)


# ==========================================================
# 5. SELECT FEATURES AND TARGET
# ==========================================================

X = data[
    [
        "Energy_Consumption",
        "Renewable_Percentage",
        "GDP"
    ]
]

y = data["CO2_Emissions"]


# ==========================================================
# 6. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================================
# 7. CREATE POLYNOMIAL REGRESSION PIPELINE
# ==========================================================

model = Pipeline(
    steps=[
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "linear_regression",
            LinearRegression()
        )
    ]
)


# ==========================================================
# 8. TRAIN MODEL
# ==========================================================

print("\nTraining Polynomial Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================================
# 9. PREDICT TEST DATA
# ==========================================================

y_pred = model.predict(X_test)


# ==========================================================
# 10. MODEL EVALUATION
# ==========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("===================================")


# ==========================================================
# 11. CREATE MODEL DIRECTORY
# ==========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# 12. SAVE MODEL
# ==========================================================

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)


# ==========================================================
# 13. SAVE TEST RESULTS
# ==========================================================

results = pd.DataFrame({
    "Actual_CO2_Emissions": y_test.values,
    "Predicted_CO2_Emissions": y_pred
})

results_path = os.path.join(
    MODEL_DIR,
    "test_results.csv"
)

results.to_csv(results_path, index=False)

print("Test results saved to:", results_path)

print("\nTraining process completed successfully!")