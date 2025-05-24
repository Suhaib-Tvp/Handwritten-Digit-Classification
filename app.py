import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import joblib
from utils import preprocess_image, predict_digit

st.title("MNIST Digit Classifier (Multinomial Logistic Regression)")

st.subheader("Draw a digit (0-9) below:")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

st.subheader("Or upload a 28x28 grayscale image:")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Load model
model = joblib.load("mnist_logreg.pkl")

image_to_predict = None

if canvas_result.image_data is not None and np.max(canvas_result.image_data) > 0:
    img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
    image_to_predict = preprocess_image(img)
elif uploaded_file is not None:
    img = Image.open(uploaded_file)
    image_to_predict = preprocess_image(img)

if image_to_predict is not None:
    pred_label, confidences = predict_digit(model, image_to_predict)
    st.write(f"**Predicted Digit:** {pred_label}")
    st.write("**Confidence Scores:**")
    for i, score in enumerate(confidences):
        st.write(f"{i}: {score:.4f}")
else:
    st.info("Draw a digit or upload an image to see predictions.")
