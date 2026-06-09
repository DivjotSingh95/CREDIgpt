import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Add project directory to sys.path
project_dir = Path("d:/data science proj/home-credit-default-risk")
sys.path.insert(0, str(project_dir))

from app import load_model, run_predictions

def main():
    print("Updating cached stats with calibrated predictions...")
    
    # 1. Load the model
    model = load_model()
    if model is None:
        print("Error: XGBoost model not found.")
        return
        
    # 2. Check for engineered features
    data_path = project_dir / "database" / "engineered_features.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Running fallback initialization...")
        # Fall back to application_test.csv sample if no uploaded data exists yet
        test_path = project_dir / "application_test.csv"
        if not test_path.exists():
            print("Error: application_test.csv not found either.")
            return
        df = pd.read_csv(test_path, nrows=500)  # load a larger sample for fallback
        from utils.mapping import get_base_target_columns, get_heuristic_mappings
        from utils.validation import run_feature_engineering_pipeline
        base_targets = get_base_target_columns()
        full_mappings = get_heuristic_mappings(list(df.columns), base_targets, {})
        confirmed_mappings = {col: full_mappings[col]["model_feature"] for col in df.columns}
        _, _, df = run_feature_engineering_pipeline(df, confirmed_mappings)
    else:
        print(f"Loading {data_path}...")
        df = pd.read_csv(data_path)
        
    # 3. Run predictions
    print(f"Running predictions on {len(df)} rows...")
    stats = run_predictions(df, model)
    
    # 4. Save to dashboard_stats.json
    db_dir = project_dir / "database"
    os.makedirs(db_dir, exist_ok=True)
    stats_path = db_dir / "dashboard_stats.json"
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Success! Stats persisted to {stats_path}")
    print(f"Total customers: {stats['total_customers']}")
    print(f"Default rate: {stats['default_rate']:.4%}")
    print(f"Avg risk score: {stats['avg_risk_score']:.4%}")
    print(f"Low risk count: {stats['low_risk_count']}")
    print(f"Medium risk count: {stats['medium_risk_count']}")
    print(f"High risk count: {stats['high_risk_count']}")
    print(f"Critical count: {stats['critical_count']}")

if __name__ == "__main__":
    main()
