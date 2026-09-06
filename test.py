# test.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


# ==========================================================
# 1. FILE PATHS
# ==========================================================

DATA_PATH = "sustainability_data.csv"

MODEL_PATH = os.path.join(
    "model",
    "co2_emission_model.pkl"
)


# ==========================================================
# 2. CHECK MODEL EXISTS
# ==========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found!\n"
        "Please run train.py first."
    )


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("\nDataset shape:", data.shape)


# ==========================================================
# 4. SELECT FEATURES AND TARGET
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
# 5. REMOVE MISSING VALUES
# ==========================================================

valid_data = pd.concat(
    [X, y],
    axis=1
).dropna()

X = valid_data[
    [
        "Energy_Consumption",
        "Renewable_Percentage",
        "GDP"
    ]
]

y = valid_data["CO2_Emissions"]


# ==========================================================
# 6. CREATE SAME TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================================
# 7. LOAD TRAINED MODEL
# ==========================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================================
# 8. MAKE PREDICTIONS
# ==========================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("Predictions completed!")


# ==========================================================
# 9. CALCULATE PERFORMANCE
# ==========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================================
# 10. DISPLAY PERFORMANCE
# ==========================================================

print("\n===================================")
print("       TEST RESULTS")
print("===================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("===================================")


# ==========================================================
# 11. DISPLAY ACTUAL VS PREDICTED
# ==========================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(20))


# ==========================================================
# 12. PLOT ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual CO2 Emissions")

plt.ylabel("Predicted CO2 Emissions")

plt.title(
    "Actual vs Predicted CO2 Emissions"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ==========================================================
# 13. SAVE TEST RESULTS
# ==========================================================

os.makedirs("model", exist_ok=True)

results_path = os.path.join(
    "model",
    "test_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nTest results saved to:")
print(results_path)

print("\nTesting completed successfully!")