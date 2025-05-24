# utils.py
import cv2
import numpy as np

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = 255 - img
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img = img / 255.0
    img = img.reshape(1, -1).astype(np.float32)
    return img
