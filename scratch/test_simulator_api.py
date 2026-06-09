import requests
import json
import os

def test_simulator_for_type(dataset_type):
    print(f"\n=========================================")
    print(f"Testing Simulator for Dataset Type: {dataset_type}")
    print(f"=========================================")
    base_url = "http://127.0.0.1:8000"
    
    # 1. Load sample dataset
    print(f"1. Loading sample dataset ({dataset_type})...")
    res = requests.get(f"{base_url}/api/load-sample?type={dataset_type}")
    if res.status_code != 200:
        print("Failed to load sample:", res.text)
        return False
    
    data = res.json()
    mappings = data.get("mappings", {})
    file_name = data.get("file_name")
    print(f"   Loaded: {file_name}, Columns: {len(data.get('columns', []))}")
    
    # 2. Process the mappings
    print("2. Processing mappings to compute baseline predictions...")
    confirm_mappings = {}
    for col, map_info in mappings.items():
        confirm_mappings[col] = map_info.get("model_feature")
        
    process_payload = {"mappings": confirm_mappings}
    res = requests.post(f"{base_url}/api/process", json=process_payload)
    if res.status_code != 200:
        print("Failed to process mappings:", res.text)
        return False
        
    process_data = res.json()
    stats = process_data.get("stats", {})
    profiles = stats.get("customer_profiles", [])
    if not profiles:
        print("Error: No customer profiles returned in stats.")
        return False
        
    target_profile = profiles[0]
    print(f"   Baseline customer profile:")
    print(f"     ID: {target_profile['id']}")
    print(f"     Income: ${target_profile['income']:,.2f}")
    print(f"     Credit: ${target_profile['credit']:,.2f}")
    print(f"     Age: {target_profile['age']:.2f} years")
    print(f"     Employment: {target_profile['employment']:.2f} years")
    print(f"     EXT Scores: [{target_profile.get('ext1')}, {target_profile.get('ext2')}, {target_profile.get('ext3')}]")
    print(f"     PD (Probability of Default): {target_profile['pd']*100:.2f}% (Risk: {target_profile['risk_level']})")

    # 3. Run simulator with modified values
    # Let's simulate reducing risk (increasing credit scores, lowering credit-to-income, etc.)
    print("\n3. Simulating modified risk values (Scenario A: Lower Risk)...")
    sim_payload = {
        "customer_id": target_profile["id"],
        "income": target_profile["income"] * 1.5, # 50% more income
        "credit": target_profile["credit"] * 0.8,  # 20% less credit
        "age": target_profile["age"],
        "employment": target_profile["employment"] + 2.0, # 2 more years of tenure
        "ext1": 0.85, # excellent rating
        "ext2": 0.90, # excellent rating
        "ext3": 0.80, # excellent rating
        "mappings": confirm_mappings
    }
    
    res = requests.post(f"{base_url}/api/simulate", json=sim_payload)
    if res.status_code != 200:
        print("Simulation A failed:", res.text)
        return False
        
    sim_data = res.json()
    if not sim_data.get("success") or not sim_data.get("profile"):
        print("Simulation A response was unsuccessful:", sim_data)
        return False
        
    sim_profile = sim_data["profile"]
    print(f"   Simulated profile (Lower Risk):")
    print(f"     ID: {sim_profile['id']}")
    print(f"     Income: ${sim_profile['income']:,.2f}")
    print(f"     Credit: ${sim_profile['credit']:,.2f}")
    print(f"     Age: {sim_profile['age']:.2f} years")
    print(f"     Employment: {sim_profile['employment']:.2f} years")
    print(f"     EXT Scores: [{sim_profile.get('ext1')}, {sim_profile.get('ext2')}, {sim_profile.get('ext3')}]")
    print(f"     PD (Probability of Default): {sim_profile['pd']*100:.2f}% (Risk: {sim_profile['risk_level']})")
    
    # 4. Run simulator with highly critical values
    print("\n4. Simulating modified risk values (Scenario B: Higher/Critical Risk)...")
    sim_payload_high = {
        "customer_id": target_profile["id"],
        "income": target_profile["income"] * 0.5, # 50% less income
        "credit": target_profile["credit"] * 2.0,  # 2x credit
        "age": target_profile["age"],
        "employment": 0.5, # very short tenure
        "ext1": 0.05, # very poor rating
        "ext2": 0.05, # very poor rating
        "ext3": 0.05, # very poor rating
        "mappings": confirm_mappings
    }
    
    res = requests.post(f"{base_url}/api/simulate", json=sim_payload_high)
    if res.status_code != 200:
        print("Simulation B failed:", res.text)
        return False
        
    sim_data_high = res.json()
    if not sim_data_high.get("success") or not sim_data_high.get("profile"):
        print("Simulation B response was unsuccessful:", sim_data_high)
        return False
        
    sim_profile_high = sim_data_high["profile"]
    print(f"   Simulated profile (Higher Risk):")
    print(f"     ID: {sim_profile_high['id']}")
    print(f"     Income: ${sim_profile_high['income']:,.2f}")
    print(f"     Credit: ${sim_profile_high['credit']:,.2f}")
    print(f"     Age: {sim_profile_high['age']:.2f} years")
    print(f"     Employment: {sim_profile_high['employment']:.2f} years")
    print(f"     EXT Scores: [{sim_profile_high.get('ext1')}, {sim_profile_high.get('ext2')}, {sim_profile_high.get('ext3')}]")
    print(f"     PD (Probability of Default): {sim_profile_high['pd']*100:.2f}% (Risk: {sim_profile_high['risk_level']})")
    
    # Assertions / Checks
    # The PD in Scenario B should be significantly higher than Scenario A
    if sim_profile_high['pd'] <= sim_profile['pd']:
        print("Error: High risk simulation did not increase probability of default!")
        return False
        
    print(f"\nAll tests passed for dataset type: {dataset_type}!")
    return True

def main():
    success_portfolio = test_simulator_for_type("portfolio")
    success_test = test_simulator_for_type("test")
    
    if success_portfolio and success_test:
        print("\n=========================================")
        print("SUCCESS: Simulator API fully validated!")
        print("=========================================")
    else:
        print("\n=========================================")
        print("FAILURE: Simulator API checks failed.")
        print("=========================================")

if __name__ == "__main__":
    main()
