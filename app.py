import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from utils import preprocess_image, predict_digit
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("mnist_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'mnist_model.pkl' not found. Please ensure the model is trained and saved.")
        return None

def main():
    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("Upload an image of a handwritten digit (0-9) and let AI predict what it is!")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a handwritten digit"
        )
        
        # Drawing canvas option
        st.markdown("---")
        st.subheader("Or draw a digit:")
        
        # Simple drawing instructions
        st.info("💡 **Tip**: For best results, draw a large, clear digit with thick lines on a white background")
        
    with col2:
        st.header("🔮 Prediction Results")
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=200)
                
                # Convert PIL image to OpenCV format for processing
                img_array = np.array(image)
                
                # Preprocess the image
                processed_img = preprocess_image(img_array)
                
                # Make prediction
                prediction, confidence = predict_digit(model, processed_img)
                
                # Display results
                st.success(f"**Predicted Digit: {prediction}**")
                st.info(f"**Confidence: {confidence:.2%}**")
                
                # Show processed image
                st.subheader("Processed Image (28x28)")
                processed_display = processed_img.reshape(28, 28)
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(processed_display, cmap='gray')
                ax.set_title("Preprocessed for Model")
                ax.axis('off')
                st.pyplot(fig)
                
                # Show confidence for all digits
                st.subheader("Confidence for All Digits")
                probabilities = model.predict_proba(processed_img.reshape(1, -1))[0]
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                digits = range(10)
                bars = ax.bar(digits, probabilities)
                
                # Highlight the predicted digit
                bars[prediction].set_color('red')
                
                ax.set_xlabel('Digit')
                ax.set_ylabel('Confidence')
                ax.set_title('Prediction Confidence for Each Digit')
                ax.set_xticks(digits)
                
                # Add percentage labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2%}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try uploading a different image or check the image format.")
        
        else:
            st.info("👆 Upload an image to see the prediction")
    
    # Instructions section
    st.markdown("---")
    st.header("📋 Instructions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("✅ Good Images")
        st.markdown("""
        - Clear, single digits
        - Good contrast (dark digit on light background)
        - Minimal noise/background
        - Centered digit
        """)
    
    with col2:
        st.subheader("❌ Avoid")
        st.markdown("""
        - Multiple digits
        - Very blurry images
        - Too much background noise
        - Extremely small digits
        """)
    
    with col3:
        st.subheader("🔧 Image Processing")
        st.markdown("""
        - Images are resized to 28x28 pixels
        - Converted to grayscale
        - Normalized for the model
        - Thresholded for clarity
        """)
    
    # Model information
    st.markdown("---")
    st.header("🤖 Model Information")
    st.markdown("""
    This application uses a **Logistic Regression** model trained on the **MNIST dataset**.
    
    - **Dataset**: 60,000 training images of handwritten digits
    - **Model**: Multinomial Logistic Regression
    - **Accuracy**: ~92% on test set
    - **Input**: 28x28 grayscale images (784 features)
    - **Output**: Probability distribution over digits 0-9
    """)

if __name__ == "__main__":
    main()
