import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from utils import preprocess_image, load_model, predict_digit

st.set_page_config(page_title="Handwritten Digit Detection", page_icon="✍️")
st.title("Handwritten Digit Detection")

st.subheader("Draw a digit (0-9):")
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

model = load_model()

image_to_predict = None

if canvas_result.image_data is not None and np.max(canvas_result.image_data) > 0:
    img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
    image_to_predict = preprocess_image(img)
elif uploaded_file is not None:
    img = Image.open(uploaded_file)
    image_to_predict = preprocess_image(img)

if image_to_predict is not None:
    pred_label = predict_digit(model, image_to_predict)
    st.markdown(
        f"""
        <div style="text-align:center;">
            <h2 style="color:#0072C6;">Detected Digit: <span style="font-size:2.5em;">{pred_label}</span></h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Draw a digit or upload an image to detect it.")
