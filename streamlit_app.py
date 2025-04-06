import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('pneumonia_detector_model.h5')

# Set title
st.title("🩻 Pneumonia Detection from Chest X-rays")
st.markdown("Upload a chest X-ray image and get a prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Display result
    if prediction < 0.5:
        st.success("✅ Prediction: **Normal**")
    else:
        st.error("⚠️ Prediction: **Pneumonia suspected**")
        st.markdown("*Please consult a medical professional for accurate diagnosis.*")

    if prediction < 0.5:
        st.success("✅ Prediction: **Normal**")
    else:
        st.error("⚠️ Prediction: **Pneumonia suspected**")
        st.markdown("*Please consult a medical professional.*")
