import cv2
import numpy as np

def preprocess_image(image_file):
    """Preprocess uploaded image file for MNIST model prediction."""
    # Read image as grayscale
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert colors (MNIST: white digit on black background)
    img = 255 - img
    # Threshold to clean image
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # Normalize
    img = img / 255.0
    # Flatten
    img = img.reshape(1, -1).astype(np.float32)
    return img
