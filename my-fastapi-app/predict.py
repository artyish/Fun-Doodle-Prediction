from tensorflow.keras.models import load_model
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

encoder = joblib.load('my-fastapi-app/label_encoder.joblib')
model_path = os.path.join(os.path.dirname(__file__), "main_model.h5")
model = load_model(model_path)

def predict_doodle(img_array):
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
    


