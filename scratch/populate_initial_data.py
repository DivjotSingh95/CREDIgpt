import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project directory to sys.path
project_dir = Path("d:/data science proj/home-credit-default-risk")
sys.path.insert(0, str(project_dir))

from utils.mapping import get_base_target_columns, get_heuristic_mappings
from utils.validation import run_feature_engineering_pipeline
from app import load_model, run_predictions

def populate_initial_data():
    print("Populating initial dashboard data...")

    # 1. Load application test sample
    test_path = project_dir / "application_train.csv"
    if not test_path.exists():
        test_path = project_dir / "application_test.csv"
    if not test_path.exists():
        print(f"Error: Neither application_train.csv nor application_test.csv found.")
        return

    print(f"Loading {test_path.name} sample...")
    df = pd.read_csv(test_path, nrows=50)

    # 2. Map columns using heuristics
    base_targets = get_base_target_columns()
    full_mappings = get_heuristic_mappings(list(df.columns), base_targets, {})
    confirmed_mappings = {col: full_mappings[col]["model_feature"] for col in df.columns}

    # 3. Run feature engineering pipeline
    print("Engineering features...")
    original, mapped, engineered = run_feature_engineering_pipeline(df, confirmed_mappings)

    # 4. Load model and run predictions
    print("Running predictions...")
    model = load_model()
    if model is None:
        print("Error: XGBoost model not found.")
        return

    stats = run_predictions(engineered, model)

    # 5. Persist to database/dashboard_stats.json
    db_dir = project_dir / "database"
    os.makedirs(db_dir, exist_ok=True)
    stats_path = db_dir / "dashboard_stats.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Success! Initial data populated. Saved to {stats_path}")

if __name__ == "__main__":
    populate_initial_data()
