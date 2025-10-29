import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🩺 Pneumonia Detection from Chest X-Ray")

model = tf.keras.models.load_model('pneumonia_model.h5')

uploaded_file = st.file_uploader("Upload a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    st.write("Prediction score:", prediction)

    if prediction > 0.5:
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal")
