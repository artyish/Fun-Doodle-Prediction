from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

def get_model():
    global _model
    if _model is None:
        print("Loading model lazily...")
        _model = tf.keras.models.load_model("main_model")
    return _model

def predict_doodle(img_array):
    encoder = joblib.load('label_encoder.joblib')
    model = get_model()
    img_resized = cv2.resize(img_array, (96, 96))
    img_normalized = img_resized / 255.0
    img_array_to_use = img_normalized.reshape(1, 96, 96, 1)

    prediction = model.predict(img_array_to_use)[0] # because our predictions our actually returned as (1 , number_of_classes)
    prediction_sorted = np.argsort(prediction)[::-1][:3] # pick last 3 using argsort / argsort actually returns the index instead of the prediction which is pretty cool
    print(prediction_sorted)
    top_3_probabilities = []
    for i in prediction_sorted:
        label = encoder.inverse_transform([i])[0]
        probability = prediction[i] * 100
        top_3_probabilities.append((label, probability))

    return top_3_probabilities
    


