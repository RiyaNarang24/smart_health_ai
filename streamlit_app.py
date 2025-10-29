import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

st.set_page_config(page_title="Smart Health AI - Pneumonia Detection")

st.title("🩺 Smart Health AI")
st.write("Upload a chest X-ray image to detect Pneumonia using a trained CNN model.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("disease_model.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    img_path = os.path.join("temp_image.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

    st.subheader("Prediction:")
    st.write(f"🧠 The model predicts: **{result}**")

st.caption("Developed by Riya Narang 💡")
