import pandas as pd
import os
import sys

# Add project directory to sys.path
sys.path.insert(0, os.getcwd())

from app import load_model, run_predictions
from utils.mapping import get_base_target_columns, get_heuristic_mappings
from utils.validation import run_feature_engineering_pipeline

def check_distribution(file_path, name):
    print(f"\nAnalyzing distribution for: {name} ({file_path})")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return
        
    df = pd.read_csv(file_path)
    print(f"  Total records in file: {len(df)}")
    
    # Map headers to model format
    base_targets = get_base_target_columns()
    full_mappings = get_heuristic_mappings(list(df.columns), base_targets, {})
    confirmed_mappings = {col: full_mappings[col]["model_feature"] for col in df.columns}
    
    # Process and engineer
    _, _, engineered = run_feature_engineering_pipeline(df, confirmed_mappings)
    
    # Predict
    model = load_model()
    if model is None:
        print("Error: Model not found.")
        return
        
    stats = run_predictions(engineered, model)
    
    print("  Risk Band Distribution:")
    print(f"    Low Risk (< 7%): {stats.get('low_risk_count')} ({stats.get('low_risk_count')/len(df)*100:.1f}%)")
    print(f"    Medium Risk (7% - 15%): {stats.get('medium_risk_count')} ({stats.get('medium_risk_count')/len(df)*100:.1f}%)")
    print(f"    High Risk (>= 15%): {stats.get('high_risk_count')} ({stats.get('high_risk_count')/len(df)*100:.1f}%)")
    print(f"    Critical Cases (>= 25%): {stats.get('critical_count')} ({stats.get('critical_count')/len(df)*100:.1f}%)")
    print(f"    Average Portfolio PD: {stats.get('avg_risk_score')*100:.2f}%")
    print(f"    Expected Default Rate: {stats.get('default_rate')*100:.2f}%")

def main():
    check_distribution("scratch/sample_portfolio_upload.csv", "Sample Portfolio")
    check_distribution("scratch/sample_train_upload.csv", "Sample Train Dataset")

if __name__ == "__main__":
    main()
