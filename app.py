import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from utils import (
    load_model_and_scaler, 
    preprocess_uploaded_image, 
    preprocess_canvas_image,
    predict_digit,
    display_prediction_results,
    center_digit_in_image,
    apply_noise_reduction
)

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title
    st.markdown('<h1 class="main-header">🔢 Handwritten Digit Classifier</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("❌ Failed to load model. Please run train_model.py first!")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        ### How to use:
        1. **Draw** a digit on the canvas, OR
        2. **Upload** an image of a handwritten digit
        3. Click **Predict** to get the classification result
        
        ### Tips for better accuracy:
        - Draw digits clearly and centered
        - Use thick strokes
        - Keep digits similar to standard handwriting
        - For uploads: use images with dark digits on light background
        """)
        
        st.header("⚙️ Settings")
        drawing_mode = st.selectbox("Drawing mode:", ["freedraw", "line", "rect", "circle"])
        stroke_width = st.slider("Stroke width:", 1, 25, 15)
        
        # Model info
        st.header("🤖 Model Information")
        st.info("""
        **Algorithm:** Logistic Regression
        **Classes:** 10 digits (0-9)
        **Input:** 28×28 grayscale images
        **Features:** 784 pixels
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["🎨 Draw Digit", "📁 Upload Image"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Draw a digit on the canvas</h2>', unsafe_allow_html=True)
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Canvas for drawing
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=stroke_width,
                stroke_color="white",
                background_color="black",
                background_image=None,
                update_streamlit=True,
                height=400,
                width=400,
                drawing_mode=drawing_mode,
                point_display_radius=0,
                display_toolbar=True,
                key="canvas"
            )
            
            # Buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_canvas = st.button("🔮 Predict Canvas", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🗑️ Clear Canvas", use_container_width=True):
                    st.rerun()
        
        with col2:
            if canvas_result.image_data is not None:
                # Show what the model will see
                st.subheader("Model Input Preview")
                
                # Process canvas image
                canvas_image = preprocess_canvas_image(canvas_result)
                
                if canvas_image is not None:
                    # Apply preprocessing
                    processed_image = center_digit_in_image(canvas_image)
                    processed_image = apply_noise_reduction(processed_image)
                    
                    # Display processed image
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                    
                    ax1.imshow(canvas_image, cmap='gray')
                    ax1.set_title('Raw Canvas')
                    ax1.axis('off')
                    
                    ax2.imshow(processed_image, cmap='gray')
                    ax2.set_title('Processed (28×28)')
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Prediction
                    if predict_canvas:
                        with st.spinner("Making prediction..."):
                            prediction, confidence, probabilities = predict_digit(model, scaler, processed_image)
                            
                            if prediction is not None:
                                display_prediction_results(prediction, confidence, probabilities)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Upload an image of a handwritten digit</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing a single handwritten digit"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display original image
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption='Original Image', use_container_width=True)
                
                # Predict button
                predict_upload = st.button("🔮 Predict Upload", type="primary", use_container_width=True)
            
            with col2:
                # Process and show what model sees
                st.subheader("Model Input Preview")
                
                processed_image = preprocess_uploaded_image(uploaded_file)
                
                if processed_image is not None:
                    # Apply additional preprocessing
                    processed_image = center_digit_in_image(processed_image)
                    processed_image = apply_noise_reduction(processed_image)
                    
                    # Display processed image
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.imshow(processed_image, cmap='gray')
                    ax.set_title('Processed Image (28×28)')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show pixel statistics
                    st.info(f"""
                    **Image Statistics:**
                    - Size: {processed_image.shape}
                    - Min pixel value: {np.min(processed_image)}
                    - Max pixel value: {np.max(processed_image)}
                    - Mean pixel value: {np.mean(processed_image):.1f}
                    """)
                    
                    # Prediction
                    if predict_upload:
                        with st.spinner("Making prediction..."):
                            prediction, confidence, probabilities = predict_digit(model, scaler, processed_image)
                            
                            if prediction is not None:
                                display_prediction_results(prediction, confidence, probabilities)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by Logistic Regression | Built with Streamlit</p>
        <p>Model trained on MNIST dataset with 60,000 training samples</p>
    </div>
    """, unsafe_allow_html=True)

# Add some example usage
with st.expander("💡 See Example Usage"):
    st.markdown("""
    ### Example Workflow:
    
    1. **For Drawing:**
       - Use thick strokes (stroke width 10-20)
       - Draw in the center of the canvas
       - Make digits clear and recognizable
       - Click "Predict Canvas" to get results
    
    2. **For Image Upload:**
       - Upload clear images with single digits
       - Best results with dark digits on light background
       - Images will be automatically resized to 28×28 pixels
       - Click "Predict Upload" to classify
    
    3. **Understanding Results:**
       - Main prediction shows the most likely digit
       - Confidence score indicates how certain the model is
       - Probability distribution shows likelihood for each digit (0-9)
    """)

if __name__ == "__main__":
    main()
