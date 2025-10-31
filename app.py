from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load pneumonia detection model
MODEL_PATH = 'pneumonia_model.h5'
model = load_model(MODEL_PATH)

# Define a function to make prediction
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected")

    # Save uploaded image
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Make prediction
    pred = predict_pneumonia(file_path)

    if pred > 0.5:
        result = "The X-ray indicates **Pneumonia**."
    else:
        result = "The X-ray appears **Normal**."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
