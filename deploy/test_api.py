import requests
import time

# URL from your deployment
URL = "https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1/chat/completions"

def check_status():
    print("⏳ Attempting to connect...")
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Are you online?"}],
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        response = requests.post(URL, json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status()
