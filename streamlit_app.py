import streamlit as st
import zipfile
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Download the zip file from GitHub
ZIP_URL = 'https://github.com/your-username/your-repo-name/raw/main/pneumonia_detector_model.zip'
ZIP_PATH = 'pneumonia_detector_model.zip'
MODEL_PATH = 'pneumonia_detector_model.h5'

@st.cache_resource
def download_and_extract_model():
    if not os.path.exists(MODEL_PATH):
        # Download zip
        with open(ZIP_PATH, 'wb') as f:
            response = requests.get(ZIP_URL)
            f.write(response.content)
        # Extract zip
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
    return load_model(MODEL_PATH)

model = download_and_extract_model()

# Streamlit UI
st.title("Pneumonia Detector")
st.write("Upload a chest X-ray image to detect pneumonia.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image (resize to match your model input)
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"
    
    st.subheader("Result:")
    st.success(result)
