import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model.h5')

st.title("Handwritten Digit Recognition")

# Option to upload or draw
option = st.radio("Choose input method:", ('Upload an image', 'Draw a digit'))

def preprocess(img):
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (if needed)
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)
    return img

if option == 'Upload an image':
    uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        if st.button('Predict'):
            input_img = preprocess(img)
            prediction = np.argmax(model.predict(input_img), axis=1)[0]
            st.success(f'Predicted Digit: {prediction}')

elif option == 'Draw a digit':
    st.write("Draw a digit below:")
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
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
        st.image(img, caption='Drawn Image', use_column_width=True)
        if st.button('Predict'):
            input_img = preprocess(img)
            prediction = np.argmax(model.predict(input_img), axis=1)[0]
            st.success(f'Predicted Digit: {prediction}')
