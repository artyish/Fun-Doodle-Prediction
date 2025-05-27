from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib
import gc


_model = None

def predict_doodle(img_array):
    encoder = joblib.load('label_encoder.joblib')
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_resized = cv2.resize(img_array, (96, 96))
    img_normalized = img_resized / 255.0
    img_array_to_use = img_normalized.reshape(1, 96, 96, 1)
    img_array_to_use = img_array_to_use.astype('float32')

    interpreter.set_tensor(input_details[0]['index'], img_array_to_use)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0] # because our predictions our actually returned as (1 , number_of_classes)
    prediction_sorted = np.argsort(prediction)[::-1][:3] # pick last 3 using argsort / argsort actually returns the index instead of the prediction which is pretty cool
    print(prediction_sorted)
    top_3_probabilities = []
    for i in prediction_sorted:
        label = encoder.inverse_transform([i])[0]
        probability = prediction[i] * 100
        top_3_probabilities.append((label, probability))

    del img_resized, img_normalized, img_array_to_use, prediction
    gc.collect()

    return top_3_probabilities
    


