import requests

response = requests.post(
    url='http://127.0.0.1:8001/predict',
    json="请根据给定的上下联第一个字生成完整对联：天地"
)

print(response.json())
