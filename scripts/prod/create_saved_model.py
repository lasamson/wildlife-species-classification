import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('bin/custom/custom_dropout_0.5_100_0.846_0.521.keras')
# tf.saved_model.save(model, 'bin/saved_model')
model.export('bin/saved_model')