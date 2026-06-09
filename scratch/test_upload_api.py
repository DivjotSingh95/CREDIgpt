import requests
import pandas as pd
import os

def main():
    print("Testing dataset upload and processing API...")
    base_url = "http://127.0.0.1:8000"
    
    # 1. Create a sample CSV file to upload
    test_csv_path = "application_test.csv"
    if not os.path.exists(test_csv_path):
        print(f"Error: {test_csv_path} not found.")
        return
        
    print("Creating sample upload CSV...")
    df = pd.read_csv(test_csv_path, nrows=20)
    # Rename some columns to simulate real-world uploads needing mapping
    df = df.rename(columns={
        "SK_ID_CURR": "Applicant ID",
        "AMT_INCOME_TOTAL": "Annual Earnings",
        "AMT_CREDIT": "Loan Value",
        "DAYS_BIRTH": "Age in Days",
        "DAYS_EMPLOYED": "Employment Days",
        "CODE_GENDER": "Gender Code"
    })
    
    sample_file_path = "scratch/sample_upload.csv"
    df.to_csv(sample_file_path, index=False)
    print(f"Sample file created at {sample_file_path}")
    
    # 2. Upload the raw file
    print("Uploading file to /api/upload-raw...")
    try:
        with open(sample_file_path, "rb") as f:
            files = {"file": (os.path.basename(sample_file_path), f, "text/csv")}
            res = requests.post(f"{base_url}/api/upload-raw", files=files)
            
        print("Upload Status Code:", res.status_code)
        if res.status_code != 200:
            print("Error Details:", res.text)
            return
            
        data = res.json()
        print("Upload Response keys:", list(data.keys()))
        print("Uploaded columns count:", len(data.get("columns", [])))
        print("Mappings found for:")
        for col, map_info in data.get("mappings", {}).items():
            if map_info.get("model_feature"):
                print(f"  {col} -> {map_info.get('model_feature')} (Conf: {map_info.get('confidence')})")
                
    except Exception as e:
        print("Upload endpoint test failed:", e)
        return
        
    # 3. Confirm mappings and process
    print("Confirming mappings and processing dataset via /api/process...")
    # We build the confirmation payload from the mapping response
    confirm_mappings = {}
    for col, map_info in data.get("mappings", {}).items():
        confirm_mappings[col] = map_info.get("model_feature")
        
    payload = {"mappings": confirm_mappings}
    
    try:
        res = requests.post(f"{base_url}/api/process", json=payload)
        print("Process Status Code:", res.status_code)
        if res.status_code != 200:
            print("Error Details:", res.text)
            return
            
        result = res.json()
        print("Process Response Success:", result.get("success"))
        print("Database Write Success:", result.get("db_success"))
        if not result.get("db_success"):
            print("Database Skip Reason (Expected if local Postgres is off):", result.get("db_error"))
        print("Processed Rows stored:", result.get("rows_stored"))
        print("Processed Columns count:", result.get("columns_count"))
        
        # Verify dashboard_stats.json was updated
        stats = result.get("stats", {})
        print("Updated stats Total Customers:", stats.get("total_customers"))
        print("Updated stats Default Rate:", stats.get("default_rate"))
        print("Updated stats Profiles Count:", len(stats.get("customer_profiles", [])))
        
    except Exception as e:
        print("Process endpoint test failed:", e)

if __name__ == "__main__":
    main()
