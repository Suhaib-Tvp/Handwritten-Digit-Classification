import numpy as np
import cv2
from PIL import Image

def preprocess_image(img_array):
    """
    Preprocess image for MNIST model prediction
    
    Args:
        img_array: numpy array of the image
        
    Returns:
        processed_img: flattened and normalized image ready for prediction
    """
    # Convert to grayscale if it's RGB
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array.copy()
    
    # Resize to 28x28 pixels (MNIST size)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors: MNIST has white digits on black background
    # Most user images have black digits on white background
    img = 255 - img
    
    # Apply threshold to clean up the image
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    # Normalize pixel values to 0-1 range
    img = img.astype(np.float32) / 255.0
    
    # Flatten the image to 1D array (784 features)
    img = img.reshape(-1)
    
    return img

def predict_digit(model, processed_img):
    """
    Predict digit using the trained model
    
    Args:
        model: trained sklearn model
        processed_img: preprocessed image array
        
    Returns:
        prediction: predicted digit (0-9)
        confidence: confidence score for the prediction
    """
    # Reshape for model input (1 sample, 784 features)
    img_reshaped = processed_img.reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(img_reshaped)[0]
    
    # Get confidence scores for all classes
    probabilities = model.predict_proba(img_reshaped)[0]
    
    # Get confidence for the predicted class
    confidence = probabilities[prediction]
    
    return int(prediction), float(confidence)

def validate_image(image):
    """
    Validate uploaded image
    
    Args:
        image: PIL Image object
        
    Returns:
        bool: True if image is valid, False otherwise
        str: Error message if invalid
    """
    try:
        # Check if image can be converted to array
        img_array = np.array(image)
        
        # Check minimum size
        if min(img_array.shape[:2]) < 10:
            return False, "Image is too small. Please upload a larger image."
        
        # Check if image is not empty
        if img_array.size == 0:
            return False, "Image appears to be empty."
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def enhance_image_contrast(img_array):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        img_array: numpy array of the image
        
    Returns:
        enhanced_img: contrast-enhanced image
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def preprocess_image_advanced(img_array):
    """
    Advanced preprocessing with contrast enhancement and noise reduction
    
    Args:
        img_array: numpy array of the image
        
    Returns:
        processed_img: processed image ready for prediction
    """
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array.copy()
    
    # Enhance contrast
    img = enhance_image_contrast(img)
    
    # Apply slight Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors for MNIST format
    img = 255 - img
    
    # Apply adaptive threshold for better binarization
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    
    # Normalize and flatten
    img = img.astype(np.float32) / 255.0
    img = img.reshape(-1)
    
    return img
