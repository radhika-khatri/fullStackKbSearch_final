import requests

BASE_URL = "http://localhost:8000/api/qa/static-chat"

# Test fetching conversation history
def test_fetch_history():
    response = requests.post(f"{BASE_URL}/history", json={"user_id": "123"})
    print("Fetch History Response:", response.json())

# Test sending a message
def test_send_message():
    response = requests.post(BASE_URL, json={"question": "What is AI?", "user_id": "123"})
    print("Send Message Response:", response.json())

if __name__ == "__main__":
    print("Testing Fetch History...")
    test_fetch_history()

    print("\nTesting Send Message...")
    test_send_message()
