
import numpy as np
import cv2

def preprocess_image(img):
    # Convert RGBA to grayscale
    gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    # Resize to 28x28 and invert colors
    resized = cv2.resize(gray, (28, 28))
    inverted = cv2.bitwise_not(resized)
    # Flatten the image
    flat = inverted.reshape(1, -1)
    return flat

def predict_digit(image, model, scaler):
    image_scaled = scaler.transform(image)
    prediction = model.predict(image_scaled)
    return prediction[0]
