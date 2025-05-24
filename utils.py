import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = 255 - img  # Invert to match MNIST format
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img = img / 255.0
    img = img.reshape(1, -1).astype(np.float32)
    return img

def predict_digit(image_array, model):
    return model.predict(image_array)[0]
