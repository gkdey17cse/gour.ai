import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("--- Health Check ---")
    print("Status Code:", response.status_code)
    print("Response:", response.json())

def test_chat(prompt: str):
    print(f"\n--- Testing Query: '{prompt}' ---")
    payload = {"query": prompt}
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Provider Used: {data['provider']}")
        print(f"Answer:\n{data['answer']}")
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_health()
    test_chat("What is Gour's GATE rank and achievements?")