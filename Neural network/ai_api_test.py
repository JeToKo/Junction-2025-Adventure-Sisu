import requests, json

URL = "http://127.0.0.1:5000/predict"
# Single sample
payload = {"features": [2025, 6, 1, 9, 1234, 30, 0]}
r = requests.post(URL, json=payload, timeout=5)
print("Single response:", r.status_code, r.text)

# Batch
payload = {"batch": [[2025,6,1,9,51,5.2,38],[2025,6,2,10,500,20,1]]}
r = requests.post(URL, json=payload, timeout=5)
print("Batch response:", r.status_code, r.text)