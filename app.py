import streamlit as st
import joblib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_image

# Load model
@st.cache_resource
def load_model():
    return joblib.load("mnist_model.pkl")

model = load_model()

st.title("MNIST Handwritten Digit Recognizer")

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 or larger)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_img = preprocess_image(uploaded_file)
    prediction = model.predict(processed_img)[0]
    
    st.markdown(f"### Predicted Digit: **{prediction}**")
    
    # Optional: Show processed image
    fig, ax = plt.subplots()
    ax.imshow(processed_img.reshape(28, 28), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
