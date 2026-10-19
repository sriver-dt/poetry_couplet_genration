import requests

url = r'http://127.0.0.1:8000/predict/'
data = '请根据给定的上联生成下联：上高水远'
response = requests.post(url=url, json=data)
print(response.json())
