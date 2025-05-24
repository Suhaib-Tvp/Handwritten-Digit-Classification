import streamlit as st
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from utils import preprocess_image

# Load model
model = joblib.load("mnist_model.pkl")

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
