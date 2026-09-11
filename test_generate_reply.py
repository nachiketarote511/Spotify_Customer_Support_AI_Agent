import requests
import json
import time

url = "http://localhost:8001/generate-reply"
payload = {
    "message": "My app is crashing",
    "conversation_context": ""
}
try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e)
