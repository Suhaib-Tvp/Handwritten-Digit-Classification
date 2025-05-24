import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
from PIL import Image, ImageOps

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        clf = joblib.load('logistic_regression_mnist_model.joblib')
        scaler = joblib.load('mnist_scaler.joblib')
        return clf, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run the training script first.")
        return None, None

def preprocess_image(img_data, source="canvas"):
    """
    Preprocessing for both canvas and uploaded images
    """
    if img_data is None:
        return None
    
    if source == "canvas":
        # Convert canvas data to PIL Image
        img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
        # Convert to grayscale
        img_gray = img.convert('L')
    else:
        # For uploaded images
        img = img_data
        # Convert to grayscale if not already
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img
    
    # Convert to numpy array
    img_array = np.array(img_gray)
    
    # For uploaded images, invert if background is white
    if source == "upload":
        # Check if background is predominantly white
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
    
    # Find bounding box using numpy operations
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        padding = 20
        rmin = max(0, rmin - padding)
        rmax = min(img_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_array.shape[1], cmax + padding)
        
        # Crop to bounding box
        img_cropped = img_array[rmin:rmax, cmin:cmax]
    else:
        img_cropped = img_array
    
    # Resize to 20x20 first
    img_pil = Image.fromarray(img_cropped)
    img_20x20 = img_pil.resize((20, 20), Image.Resampling.LANCZOS)
    
    # Create 28x28 image with 4-pixel border
    img_28x28 = Image.new('L', (28, 28), 0)
    img_28x28.paste(img_20x20, (4, 4))
    
    # Convert to numpy array
    final_array = np.array(img_28x28)
    
    # Normalize to 0-1 range
    final_array = final_array.astype('float64') / 255.0
    
    return final_array

def make_prediction(processed_img, clf, scaler):
    """
    Make prediction and return results
    """
    if processed_img is None:
        return None, None, None
    
    # Flatten and scale for prediction
    img_flat = processed_img.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)
    
    # Get prediction and probabilities
    prediction = clf.predict(img_scaled)[0]
    probabilities = clf.predict_proba(img_scaled)[0]
    confidence = np.max(probabilities)
    
    return prediction, probabilities, confidence

def display_results(prediction, probabilities, confidence, processed_img):
    """
    Display prediction results
    """
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.markdown("**Processed Image (28×28):**")
        st.image(processed_img, width=140, caption="Ready for prediction")
    
    with col4:
        # Display results
        st.markdown("**Prediction Results:**")
        st.markdown(f"## 🎯 **{prediction}**")
        st.markdown(f"**Confidence: {confidence:.1%}**")
        
        # Show confidence bar
        st.progress(confidence)
    
    # Detailed probability breakdown
    st.markdown("**All Probabilities:**")
    import pandas as pd
    prob_data = {f"Digit {i}": prob for i, prob in enumerate(probabilities)}
    df = pd.DataFrame(list(prob_data.items()), columns=['Digit', 'Probability'])
    
    st.bar_chart(df.set_index('Digit')['Probability'])
    
    # Additional feedback
    if confidence < 0.7:
        st.warning("Low confidence prediction. Try with a clearer image.")
    elif confidence > 0.95:
        st.success("High confidence prediction!")

# Load model
clf, scaler = load_model()

st.title('🔢 MNIST Digit Recognition')
st.markdown("### Choose your input method:")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Draw on Canvas", "Upload Image"])

if clf is not None and scaler is not None:
    
    with tab1:
        st.markdown("### Draw a digit (0-9) in the canvas below")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**Drawing Canvas:**")
            canvas_result = st_canvas(
                fill_color='#000000',
                stroke_width=12,
                stroke_color='#FFFFFF',
                background_color='#000000',
                height=280,
                width=280,
                drawing_mode='freedraw',
                key='canvas'
            )
        
        with col2:
            st.markdown("**Tips for better accuracy:**")
            st.markdown("- Draw digits **large and centered**")
            st.markdown("- Use **bold strokes**")
            st.markdown("- Keep digits **well-formed**")
            st.markdown("- Avoid touching canvas edges")
            
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.rerun()
        
        # Process canvas drawing
        if canvas_result.image_data is not None:
            if np.any(canvas_result.image_data[:, :, 3] > 0):
                processed_img = preprocess_image(canvas_result.image_data, source="canvas")
                
                if processed_img is not None:
                    st.markdown("---")
                    prediction, probabilities, confidence = make_prediction(processed_img, clf, scaler)
                    display_results(prediction, probabilities, confidence, processed_img)
            else:
                st.info(" Draw a digit on the canvas above to see the prediction!")
    
    with tab2:
        st.markdown("### Upload an image containing a digit")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing a single digit (0-9)"
        )
        
        col5, col6 = st.columns([3, 2])
        
        with col6:
            st.markdown("**Tips for uploaded images:**")
            st.markdown("- Use images with **single digits**")
            st.markdown("- **Clear, high contrast** images work best")
            st.markdown("- Both **black on white** and **white on black** work")
            st.markdown("- Avoid cluttered backgrounds")
        
        if uploaded_file is not None:
            with col5:
                # Display original uploaded image
                uploaded_img = Image.open(uploaded_file)
                st.markdown("**Original Image:**")
                st.image(uploaded_img, width=280, caption="Uploaded image")
            
            # Process uploaded image
            processed_img = preprocess_image(uploaded_img, source="upload")
            
            if processed_img is not None:
                st.markdown("---")
                prediction, probabilities, confidence = make_prediction(processed_img, clf, scaler)
                display_results(prediction, probabilities, confidence, processed_img)
        else:
            st.info("Upload an image file to see the prediction!")
    
    # Model information
    st.markdown("---")
    st.markdown("Model Information")
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("**Algorithm:** Logistic Regression")
        st.markdown("**Dataset:** MNIST (70,000 samples)")
        st.markdown("**Input:** 28×28 grayscale images")
    
    with col8:
        st.markdown("**Classes:** 10 digits (0-9)")
        st.markdown("**Preprocessing:** StandardScaler")
        st.markdown("**Regularization:** L2 penalty")

else:
    st.error("Model files not found!")
    st.markdown("Please ensure these files exist in your project:")
    st.code("logistic_regression_mnist_model.joblib")
    st.code("mnist_scaler.joblib")
