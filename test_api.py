import requests
import json
import time

# Give the server some time to start up
time.sleep(2)

url = "http://localhost:8000/support-agent"
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
