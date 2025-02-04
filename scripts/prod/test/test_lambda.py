import requests
# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://13r6t0ye17.execute-api.us-east-1.amazonaws.com/test/predict'
data = {'url': 'https://drivendata-public-assets.s3.amazonaws.com/zjff-ZJ000097.jpg'}

result = requests.post(url, json=data).json()
print(result)