import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from utils import load_model, preprocess_image, predict_digit

st.title("Handwritten Digit Recognition")

# Load model (change path to your model file if needed)
model = load_model('model.pkl')

option = st.radio("Choose input method:", ('Upload an image', 'Draw a digit'))

def display_prediction(img):
    processed = preprocess_image(img)
    digit, confidence = predict_digit(model, processed)
    st.success(f"Predicted Digit: {digit} (Confidence: {confidence:.2f})")

if option == 'Upload an image':
    uploaded = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            display_prediction(img)

else:
    st.write("Draw a digit below (black on white):")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas.image_data is not None:
        img = Image.fromarray((canvas.image_data[:, :, 0:3]).astype('uint8')).convert("L")
        st.image(img, caption="Drawn Image", use_column_width=True)
        if st.button("Predict"):
            display_prediction(img)
