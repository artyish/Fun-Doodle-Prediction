import tensorflow as tf
import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

model = tf.keras.models.load_model("my-fastapi-app/main_model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)