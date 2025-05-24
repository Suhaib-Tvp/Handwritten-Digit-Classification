import numpy as np
from PIL import Image, ImageOps

def preprocess_image(img):
    # Convert to grayscale, resize, invert, binarize, flatten
    img = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    img = np.array(img)
    img = (img > 127).astype(np.float32)
    img = img.reshape(1, -1)
    return img

def predict_digit(model, processed_img):
    probs = model.predict_proba(processed_img)[0]
    pred = np.argmax(probs)
    return pred, probs
