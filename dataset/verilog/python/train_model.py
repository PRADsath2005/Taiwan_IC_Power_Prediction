import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Get project path
base_path = os.path.dirname(os.path.dirname(__file__))

# Dataset path
dataset_path = os.path.join(
    base_path,
    "dataset",
    "power_delay_dataset.csv"
)

# Load dataset
df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print(df.head())

# Input features
features = [
    "A",
    "B",
    "Cin",
    "Sum",
    "Cout",
    "Switching_Activity",
    "Transistor_Count",
    "Delay_ns"
]

X = df[features]

# Target
y = df["Power_mW"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL TRAINING COMPLETED!")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Create results folder
results_path = os.path.join(base_path, "results")
os.makedirs(results_path, exist_ok=True)

# Save model
model_path = os.path.join(results_path, "power_prediction_model.pkl")
joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(model_path)
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# ACTUAL vs PREDICTED GRAPH
# -------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Power (mW)")
plt.ylabel("Predicted Power (mW)")
plt.title("Actual vs Predicted Power")

# Perfect prediction line
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.grid(True)

actual_predicted_path = os.path.join(
    results_path,
    "actual_vs_predicted.png"
)

plt.savefig(actual_predicted_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nActual vs Predicted graph saved!")
print(actual_predicted_path)


# -------------------------------
# FEATURE IMPORTANCE GRAPH
# -------------------------------

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - IC Power Prediction")

plt.xticks(rotation=45)

plt.tight_layout()

feature_graph_path = os.path.join(
    results_path,
    "feature_importance.png"
)

plt.savefig(feature_graph_path, dpi=300)
plt.close()

print("\nFeature importance graph saved!")
print(feature_graph_path)


# -------------------------------
# SAVE PREDICTION RESULTS
# -------------------------------

prediction_results = pd.DataFrame({
    "Actual_Power_mW": y_test.values,
    "Predicted_Power_mW": predictions,
    "Error": np.abs(y_test.values - predictions)
})

prediction_csv_path = os.path.join(
    results_path,
    "prediction_results.csv"
)

prediction_results.to_csv(
    prediction_csv_path,
    index=False
)

print("\nPrediction results saved!")
print(prediction_csv_path)

print("\n" + "=" * 60)
print("TAIWAN IC POWER PREDICTION PROJECT COMPLETED!")
print("=" * 60)