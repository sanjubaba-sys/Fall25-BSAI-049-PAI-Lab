from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("model/emotion_model.h5")

# Emotion classes
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']

    # Make upload directory if not exists
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded image
    img_path = os.path.join(upload_dir, file.filename)
    file.save(img_path)

    # Read and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return "Invalid image format"

    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    prediction = model.predict(img)
    label = emotion_labels[np.argmax(prediction)]

    # Send result to result page
    return render_template("result.html", emotion=label, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
