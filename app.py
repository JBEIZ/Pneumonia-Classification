import streamlit as st
import zipfile
import requests
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_URL = "https://github.com/yourusername/yourrepo/raw/main/pneumonia_detector_model.zip"
MODEL_ZIP = "pneumonia_detector_model.zip"
MODEL_FILE = "pneumonia_detector_model.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_FILE):
        # Download the zip file
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_ZIP, 'wb') as f:
                f.write(r.content)
        # Unzip it
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
    return load_model(MODEL_FILE)

model = download_and_load_model()

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image (preferably grayscale or RGB)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.resize((150, 150))
    img = img.convert("RGB")  # Ensure 3 channels
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0][0]

    label = "Pneumonia" if prediction >= 0.5 else "Normal"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")
