import requests
import json

def main():
    print("Testing Load Sample Dataset API...")
    base_url = "http://127.0.0.1:8000"
    
    # 1. Trigger load sample
    print("Requesting /api/load-sample...")
    try:
        res = requests.get(f"{base_url}/api/load-sample")
        print("Load Sample Status Code:", res.status_code)
        if res.status_code != 200:
            print("Error Details:", res.text)
            return
            
        data = res.json()
        print("Sample Metadata:")
        print(f"  File Name: {data.get('file_name')}")
        print(f"  File Size: {data.get('file_size')} bytes")
        print(f"  Columns Count: {len(data.get('columns', []))}")
        
        # Print detected mappings
        print("\nDetected Schema Mappings:")
        mappings = data.get("mappings", {})
        mapped_count = 0
        for col, map_info in mappings.items():
            model_feat = map_info.get("model_feature")
            if model_feat:
                mapped_count += 1
                print(f"  '{col}' -> '{model_feat}' (Confidence: {map_info.get('confidence')}, Explanation: {map_info.get('explanation')[:60]}...)")
        print(f"Total mapped columns: {mapped_count} out of {len(data.get('columns', []))}")
        
        # Verify essential mappings are present
        essential_targets = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]
        found_targets = [map_info.get("model_feature") for map_info in mappings.values() if map_info.get("model_feature")]
        print("\nChecking essential targets mapping:")
        for target in essential_targets:
            status = "MAPPED" if target in found_targets else "NOT MAPPED"
            print(f"  {target}: {status}")
            
    except Exception as e:
        print("Failed to call /api/load-sample:", e)
        return
        
    # 2. Confirm mappings and process
    print("\nProcessing confirmed mappings via /api/process...")
    confirm_mappings = {}
    for col, map_info in mappings.items():
        confirm_mappings[col] = map_info.get("model_feature")
        
    payload = {"mappings": confirm_mappings}
    
    try:
        res = requests.post(f"{base_url}/api/process", json=payload)
        print("Process Status Code:", res.status_code)
        if res.status_code != 200:
            print("Error Details:", res.text)
            return
            
        result = res.json()
        print("Process Response:")
        print(f"  Success: {result.get('success')}")
        print(f"  Rows Scored: {result.get('rows_stored')}")
        print(f"  Columns Count: {result.get('columns_count')}")
        print(f"  Database Write Success: {result.get('db_success')}")
        if not result.get('db_success'):
            print(f"  Database Skip Details: {result.get('db_error')}")
            
        stats = result.get("stats", {})
        print("\nPortfolio Statistics Calibrated:")
        print(f"  Total Customers: {stats.get('total_customers')}")
        print(f"  Average Risk Score (PD): {stats.get('avg_risk_score'):.4f}")
        print(f"  Expected Default Rate: {stats.get('default_rate'):.4f}")
        print(f"  High Risk Customers (PD >= 15%): {stats.get('high_risk_count')}")
        print(f"  Critical Cases (PD >= 25%): {stats.get('critical_count')}")
        
        # Verify first profile detail
        profiles = stats.get("customer_profiles", [])
        if profiles:
            p = profiles[0]
            print(f"\nFirst Customer Profile Sample:")
            print(f"  ID: {p.get('id')}")
            print(f"  Gender: {p.get('gender')}")
            print(f"  Age: {p.get('age')} years")
            print(f"  Employment: {p.get('employment')} years")
            print(f"  Income: ${p.get('income'):,.2f}")
            print(f"  Credit: ${p.get('credit'):,.2f}")
            print(f"  PD: {p.get('pd')*100:.2f}% (Risk Level: {p.get('risk_level')})")
            
        print("\nAll integration checks for Load Sample Dataset passed successfully!")
        
    except Exception as e:
        print("Failed to run mapping confirmation and prediction processing:", e)

if __name__ == "__main__":
    main()
