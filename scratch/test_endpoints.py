import requests

def main():
    print("Testing FastAPI endpoints...")
    base_url = "http://127.0.0.1:8000"
    
    # 1. Test root page
    try:
        res = requests.get(f"{base_url}/")
        print(f"GET /: Status {res.status_code}, Length: {len(res.text)}")
    except Exception as e:
        print(f"GET / failed: {e}")
        
    # 2. Test config endpoint
    try:
        res = requests.get(f"{base_url}/api/config")
        print(f"GET /api/config: Status {res.status_code}")
        print("Response:", res.json())
    except Exception as e:
        print(f"GET /api/config failed: {e}")
        
    # 3. Test stats endpoint
    try:
        res = requests.get(f"{base_url}/api/stats")
        print(f"GET /api/stats: Status {res.status_code}")
        data = res.json()
        print("Total customers in stats:", data.get("total_customers"))
        print("Default rate in stats:", data.get("default_rate"))
        print("Profiles count:", len(data.get("customer_profiles", [])))
    except Exception as e:
        print(f"GET /api/stats failed: {e}")

if __name__ == "__main__":
    main()
