from PIL import Image
import requests
import numpy as np
from tensorflow import keras

# Output classes in the order output by classifier
classes = [
 'antelope_duiker',
 'bird',
 'blank',
 'civet_genet',
 'hog',
 'leopard',
 'monkey_prosimian',
 'rodent'
]

# Load our final/best model
model_file = 'custom_dropout_0.5_100_0.846_0.521.keras'
model = keras.models.load_model(model_file)

# Prediction function that reads image from url, and returns
# dictionary with associated softmax scores
def predict(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((150, 150), Image.NEAREST)
    X = np.array([np.array(img)])
    X = np.float32(X)

    preds = model.predict(X)
    float_preds = preds[0].tolist()
    
    return dict(zip(classes, float_preds))

# AWS Lambda function
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result