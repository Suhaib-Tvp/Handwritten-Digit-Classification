import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import io

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("mnist_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

def preprocess_image(uploaded_file):
    # Read image as grayscale
    image = Image.open(uploaded_file).convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(image)
    # Invert colors (white digit on black background)
    img_inverted = cv2.bitwise_not(img_array)
    # Flatten and scale
    img_flat = img_inverted.reshape(1, -1).astype('float32')
    img_scaled = scaler.transform(img_flat)
    return img_scaled, img_inverted

st.title("Handwritten Digit Recognition (MNIST)")

st.write("Upload an image of a single handwritten digit (0-9). The image should be clear and centered.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess and display
    try:
        img_scaled, img_show = preprocess_image(uploaded_file)
        st.image(img_show, caption='Preprocessed Image (28x28)', width=150)
        # Predict
        prediction = model.predict(img_scaled)
        st.success(f"Predicted Digit: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("---")
st.caption("Built with Streamlit · Powered by Logistic Regression (MNIST)")
