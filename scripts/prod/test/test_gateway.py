import requests

# Endpoint URL
# url = 'http://localhost:8080/predict'
url = 'http://a1e01853916d54c4abfcb6423fbe712d-1262822283.us-east-1.elb.amazonaws.com/predict'

# Request data
image_url = 'https://drivendata-public-assets.s3.amazonaws.com/zjff-ZJ000097.jpg'
data = {'url': image_url}

# Make request
result = requests.post(url, json=data).json()
print(result)