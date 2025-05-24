import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from utils import (
    load_logistic_model, load_cnn_model,
    preprocess_image_for_logistic, preprocess_image_for_cnn,
    predict_logistic, predict_cnn
)

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")
st.title("Handwritten Digit Recognition")

# Model selection
model_type = st.selectbox("Choose model:", ("Logistic Regression (.pkl)", "CNN (.h5)"))

# Load model
if model_type == "Logistic Regression (.pkl)":
    model, scaler = load_logistic_model(model_path="model.pkl", scaler_path=None)
else:
    try:
        model = load_cnn_model(model_path="cnn_model.h5")
        scaler = None
    except ImportError as e:
        st.error(str(e))
        st.stop()

# Input method
option = st.radio("Input method:", ('Upload an image', 'Draw a digit'))

def display_prediction(img):
    if model_type == "Logistic Regression (.pkl)":
        processed = preprocess_image_for_logistic(img, scaler)
        digit, confidence = predict_logistic(model, processed)
    else:
        processed = preprocess_image_for_cnn(img)
        digit, confidence = predict_cnn(model, processed)
    st.success(f"Predicted Digit: {digit} (Confidence: {confidence:.2f})")

if option == 'Upload an image':
    uploaded = st.file_uploader("Upload a digit image (28x28, black digit on white)...", type=["png", "jpg", "jpeg"])
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
