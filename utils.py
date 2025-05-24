import numpy as np
import cv2
from PIL import Image
import pickle
import streamlit as st

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        with open('digit_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run train_model.py first to generate the model files.")
        return None, None

def preprocess_image_array(image_array):
    """
    Preprocess image array for prediction
    Converts grayscale to binary and normalizes
    """
    # Ensure image is in the right format (28x28)
    if image_array.shape != (28, 28):
        # Resize if necessary
        image_array = cv2.resize(image_array, (28, 28))
    
    # Convert to binary (threshold at 127.5 for 0-255 range)
    binary_image = (image_array > 127.5).astype(np.float32)
    
    return binary_image

def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess uploaded image file
    """
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Invert if background is white (common case)
        if np.mean(image_array) > 127:
            image_array = 255 - image_array
        
        return image_array
    
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
        return None

def preprocess_canvas_image(canvas_data):
    """
    Preprocess image from canvas drawing
    """
    try:
        if canvas_data is None or canvas_data.image_data is None:
            return None
        
        # Get image data from canvas
        image_data = canvas_data.image_data
        
        # Convert to PIL Image
        image = Image.fromarray(image_data.astype('uint8'), 'RGBA')
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Invert colors (canvas draws white on black, we need black on white)
        image_array = 255 - image_array
        
        return image_array
    
    except Exception as e:
        st.error(f"Error processing canvas image: {e}")
        return None

def predict_digit(model, scaler, image_array):
    """
    Predict digit from preprocessed image array
    """
    try:
        # Preprocess image
        processed_image = preprocess_image_array(image_array)
        
        # Flatten for model input
        flattened_image = processed_image.flatten().reshape(1, -1)
        
        # Scale the image
        scaled_image = scaler.transform(flattened_image)
        
        # Make prediction
        prediction = model.predict(scaled_image)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(scaled_image)[0]
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        return prediction, confidence, probabilities
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def display_prediction_results(prediction, confidence, probabilities):
    """
    Display prediction results in a formatted way
    """
    # Main prediction
    st.success(f"🎯 **Predicted Digit: {prediction}**")
    st.info(f"📊 **Confidence Score: {confidence:.2%}**")
    
    # Confidence bar
    st.progress(confidence)
    
    # All class probabilities
    st.subheader("📈 Prediction Probabilities for All Digits")
    
    # Create columns for better display
    cols = st.columns(5)
    
    for i in range(10):
        col_idx = i % 5
        with cols[col_idx]:
            prob_percent = probabilities[i] * 100
            st.metric(
                label=f"Digit {i}",
                value=f"{prob_percent:.1f}%"
            )
    
    # Highlight top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    
    st.subheader("🏆 Top 3 Predictions")
    for i, idx in enumerate(top_3_indices):
        rank = ["🥇", "🥈", "🥉"][i]
        st.write(f"{rank} Digit {idx}: {probabilities[idx]:.2%}")

def create_sample_digit_images():
    """
    Create sample digit images for testing
    """
    samples = {}
    
    # Create simple digit patterns (28x28)
    # These are very basic representations
    
    # Digit 0
    zero = np.zeros((28, 28))
    cv2.circle(zero, (14, 14), 10, 255, 2)
    samples[0] = zero
    
    # Digit 1
    one = np.zeros((28, 28))
    cv2.line(one, (14, 5), (14, 23), 255, 2)
    cv2.line(one, (12, 7), (14, 5), 255, 2)
    samples[1] = one
    
    # Digit 2
    two = np.zeros((28, 28))
    cv2.ellipse(two, (14, 10), (8, 5), 0, 0, 180, 255, 2)
    cv2.line(two, (6, 15), (22, 23), 255, 2)
    cv2.line(two, (6, 23), (22, 23), 255, 2)
    samples[2] = two
    
    return samples

def validate_image_dimensions(image_array):
    """
    Validate and fix image dimensions
    """
    if len(image_array.shape) != 2:
        raise ValueError("Image must be 2D (grayscale)")
    
    if image_array.shape != (28, 28):
        # Resize to 28x28
        image_array = cv2.resize(image_array, (28, 28))
    
    return image_array

def normalize_pixel_values(image_array):
    """
    Ensure pixel values are in the correct range (0-255)
    """
    # Normalize to 0-255 range
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    if max_val > min_val:
        image_array = (image_array - min_val) / (max_val - min_val) * 255
    
    return image_array.astype(np.uint8)

def apply_noise_reduction(image_array):
    """
    Apply noise reduction to improve prediction accuracy
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_array, (3, 3), 0)
    
    # Apply threshold to make it more binary
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return thresh

def center_digit_in_image(image_array):
    """
    Center the digit in the 28x28 image
    """
    # Find contours
    contours, _ = cv2.findContours(image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image_array
    
    # Get bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract the digit
    digit = image_array[y:y+h, x:x+w]
    
    # Create new centered image
    centered_image = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate position to center the digit
    start_x = max(0, (28 - w) // 2)
    start_y = max(0, (28 - h) // 2)
    end_x = min(28, start_x + w)
    end_y = min(28, start_y + h)
    
    # Place digit in center
    digit_h, digit_w = digit.shape
    centered_image[start_y:start_y+digit_h, start_x:start_x+digit_w] = digit
    
    return centered_image
