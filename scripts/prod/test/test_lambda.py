import requests

# Endpoint URL
# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://13r6t0ye17.execute-api.us-east-1.amazonaws.com/test/predict'

# Request data
image_url = 'https://drivendata-public-assets.s3.amazonaws.com/zjff-ZJ000097.jpg'
data = {'url': image_url}

# Make request
result = requests.post(url, json=data).json()
print(result)