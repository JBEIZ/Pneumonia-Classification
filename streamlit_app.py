import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('pneumonia_detector_model.h5')

model = load_model()

# App UI
st.title("🩻 Pneumonia Detection from Chest X-rays")
st.markdown("Upload a chest X-ray image and get a prediction.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.success("✅ Prediction: **Normal**")
    else:
        st.error("⚠️ Prediction: **Pneumonia suspected**")
        st.markdown("*Please consult a medical professional.*")
