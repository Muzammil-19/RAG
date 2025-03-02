import requests

url = "http://127.0.0.1:8000/query/?query=\"What is the document about?\""

payload = {}
headers = {}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
