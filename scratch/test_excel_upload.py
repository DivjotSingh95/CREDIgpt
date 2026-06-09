import os
import requests
import pandas as pd

def test_excel_upload():
    csv_path = "scratch/sample_upload.csv"
    xlsx_path = "scratch/sample_upload_excel.xlsx"
    upload_filename = "sample_upload_excel.XLSX" # Use uppercase extension in the API request
    
    if not os.path.exists(csv_path):
        print("Sample CSV not found. Please run test_upload_api.py first.")
        return
        
    print(f"Converting {csv_path} to Excel format at {xlsx_path}...")
    df = pd.read_csv(csv_path)
    df.to_excel(xlsx_path, index=False)
    
    url_upload = "http://127.0.0.1:8000/api/upload-raw"
    print(f"Uploading {xlsx_path} to {url_upload} as {upload_filename}...")
    
    with open(xlsx_path, "rb") as f:
        files = {"file": (upload_filename, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        res = requests.post(url_upload, files=files)
        
    print(f"Upload Status Code: {res.status_code}")
    if res.status_code != 200:
        print(f"Failed: {res.text}")
        return
        
    data = res.json()
    print(f"Upload Successful! Columns count: {len(data['columns'])}")
    
    # Map raw columns to target features
    mappings = {col: data["mappings"][col]["model_feature"] for col in data["columns"]}
    
    url_process = "http://127.0.0.1:8000/api/process"
    print(f"Confirming mappings and processing dataset via {url_process}...")
    
    res_proc = requests.post(url_process, json={"mappings": mappings})
    print(f"Process Status Code: {res_proc.status_code}")
    if res_proc.status_code != 200:
        print(f"Failed: {res_proc.text}")
        return
        
    result = res_proc.json()
    print(f"Process Successful! Success flag: {result['success']}")
    print(f"Processed Rows stored: {result['rows_stored']}")
    print(f"Updated Stats Default Rate: {result['stats']['default_rate']*100:.2f}%")
    
    # Cleanup Excel file
    if os.path.exists(xlsx_path):
        os.remove(xlsx_path)

if __name__ == "__main__":
    test_excel_upload()
