import pandas as pd
import xgboost as xgb
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from app import encode_categoricals

MODEL_FILE = "xgboost_default_model.json"
DATA_FILE = os.path.join("database", "engineered_features.csv")

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)
df = pd.read_csv(DATA_FILE)
df_encoded = encode_categoricals(df)

expected = model.get_booster().feature_names
X = df_encoded.copy()
for col in expected:
    if col not in X.columns:
        X[col] = 0.5 if "EXT_SOURCE" in col else 0.0
X = X[expected]
for col in X.columns:
    if X[col].dtype == object:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.5 if "EXT_SOURCE" in col else 0)
for col in X.columns:
    if "EXT_SOURCE" in col or "ext" in col.lower():
        X[col] = X[col].fillna(0.5)
    else:
        X[col] = X[col].fillna(0.0)

proba = model.predict_proba(X)[:, 1]

# Calibrate probability
p_true = 0.08
p_train = 0.5
proba_cal = (proba * p_true / p_train) / (proba * p_true / p_train + (1 - proba) * (1 - p_true) / (1 - p_train))

# Predict default at High Risk threshold (>= 0.12)
pred_cal = (proba_cal >= 0.12).astype(int)

print(f"Raw - Avg PD: {proba.mean():.4f}")
print(f"Calibrated - Avg PD: {proba_cal.mean():.4f}, Default Rate (>=0.12): {pred_cal.mean():.4f}")
print(f"Low Risk Count (<5%): {(proba_cal < 0.05).sum()}")
print(f"Medium Risk Count (5-12%): {((proba_cal >= 0.05) & (proba_cal < 0.12)).sum()}")
print(f"High Risk Count (>=12%): {(proba_cal >= 0.12).sum()}")
