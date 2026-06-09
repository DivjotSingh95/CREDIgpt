import os
import google.generativeai as genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return
        
    genai.configure(api_key=api_key)
    try:
        print("Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print("Error listing models:", str(e))

if __name__ == "__main__":
    main()
