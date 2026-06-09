import requests
import json

def main():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Load sample dataset (train)
    print("Loading sample dataset...")
    res = requests.get(f"{base_url}/api/load-sample?type=train")
    data = res.json()
    mappings = {col: info.get("model_feature") for col, info in data["mappings"].items()}
    
    # 2. Process
    requests.post(f"{base_url}/api/process", json={"mappings": mappings})
    
    # 3. Simulate with extreme debt and low income
    print("\nSimulating extreme scenario:")
    print("  Annual Income: $10,000")
    print("  Credit Limit: $1,000,000")
    print("  External Ratings: [0.85, 0.90, 0.80] (Excellent)")
    
    payload = {
        "customer_id": 100002,
        "income": 10000.0,
        "credit": 1000000.0,
        "age": 40.0,
        "employment": 5.0,
        "ext1": 0.85,
        "ext2": 0.90,
        "ext3": 0.80,
        "mappings": mappings
    }
    
    res = requests.post(f"{base_url}/api/simulate", json=payload)
    result = res.json()
    
    if result.get("success"):
        p = result["profile"]
        print(f"\nSimulation Result:")
        print(f"  ID: {p['id']}")
        print(f"  Income: ${p['income']:,.2f}")
        print(f"  Credit: ${p['credit']:,.2f}")
        print(f"  PD (Probability of Default): {p['pd']*100:.2f}%")
        print(f"  Risk Level: {p['risk_level']}")
    else:
        print("Simulation failed:", result)

if __name__ == "__main__":
    main()
