
import streamlit as st
import numpy as np
import cv2
import joblib
from utils import preprocess_image, predict_digit
from PIL import Image

# Load model and scaler
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        img_array = np.array(image)
        processed = preprocess_image(img_array)
        prediction = predict_digit(processed, model, scaler)
        st.success(f"Predicted Digit: {prediction}")
