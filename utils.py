import numpy as np
from PIL import Image, ImageOps
import joblib

def load_model(model_path='model.pkl'):
    """
    Loads and returns a scikit-learn or joblib model from the given path.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    """
    Preprocesses a PIL image for digit classification:
    - Converts to grayscale
    - Inverts colors (MNIST is white digit on black)
    - Resizes to 28x28
    - Normalizes pixel values
    - Reshapes for model input
    """
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, -1)  # For sklearn models
    return img_array

def predict_digit(model, img_array):
    """
    Predicts the digit from a preprocessed image using the given model.
    Returns the predicted class (digit) and the confidence.
    """
    pred = model.predict(img_array)[0]
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(img_array))
    else:
        confidence = 1.0
    return pred, confidence
