import os
import requests

def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return
        
    print("Testing Gemini API Key with gemini-2.5-flash...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": "Hello, write a 3-word greeting."}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        if response.status_code == 200:
            print("Gemini API key is working perfectly with gemini-2.5-flash!")
        else:
            print("Gemini API key verification failed.")
    except Exception as e:
        print("Exception occurred:", str(e))

if __name__ == "__main__":
    main()
