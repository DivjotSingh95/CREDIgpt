import requests

def main():
    print("Testing chat API endpoint...")
    base_url = "http://127.0.0.1:8000"
    
    payload = {"message": "What is the predicted default rate for this portfolio?"}
    print(f"Sending question: {payload['message']}")
    
    try:
        res = requests.post(f"{base_url}/api/chat", json=payload)
        print("Chat Status Code:", res.status_code)
        if res.status_code == 200:
            print("Response:")
            print(res.json().get("response"))
        else:
            print("Error:", res.text)
    except Exception as e:
        print("Failed to query chat API:", e)

if __name__ == "__main__":
    main()
