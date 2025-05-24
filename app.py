# app.py
import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("scaler.pkl")

# Preprocessing function (no utils.py needed)
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(image)
    # Invert colors (assuming white background, black digit)
    img_array = 255 - img_array
    # Flatten and scale
    img_flat = img_array.reshape(1, -1).astype('float32')
    img_scaled = scaler.transform(img_flat)
    return img_scaled

# Title
st.title("🧠 MNIST Digit Classifier")
st.write("Upload an image of a digit (28x28 or larger), and I'll try to guess it!")

# File uploader
uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]

    st.success(f"✅ Predicted Digit: **{prediction}**")
