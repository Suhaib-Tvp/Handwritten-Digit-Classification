
import streamlit as st
import numpy as np
import cv2
import joblib
from utils import preprocess_image, predict_digit

# Load model and scaler
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the box below.")

canvas_result = st.canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    if st.button("Predict"):
        processed = preprocess_image(img)
        prediction = predict_digit(processed, model, scaler)
        st.success(f"Predicted Digit: {prediction}")
