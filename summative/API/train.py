import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Student_Performance.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.drop(["student_id", "final_grade"], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("overall_score", axis=1)
y = df["overall_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=50,   # kept small to reduce file size
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} | R²: {r2_score(y_test, y_pred):.4f}")

joblib.dump(model,  os.path.join(BASE_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

with open(os.path.join(BASE_DIR, "feature_columns.json"), "w") as f:
    json.dump(list(X.columns), f, indent=2)

print("Saved: best_model.pkl, scaler.pkl, feature_columns.json")
