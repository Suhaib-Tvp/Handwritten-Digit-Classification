import numpy as np
from PIL import Image, ImageOps
import joblib

# --- Logistic Regression (scikit-learn) functions ---

def load_logistic_model(model_path='logistic_model.pkl', scaler_path=None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler

def preprocess_image_for_logistic(img, scaler=None):
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, -1)
    if scaler:
        img_array = scaler.transform(img_array)
    return img_array

def predict_logistic(model, img_array):
    pred = model.predict(img_array)[0]
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(img_array)))
    else:
        confidence = 1.0
    return pred, confidence

# --- CNN (Keras/TensorFlow) functions ---

def load_cnn_model(model_path='cnn_model.h5'):
    try:
        from tensorflow.keras.models import load_model as keras_load_model
    except ImportError:
        raise ImportError("TensorFlow is not installed. Please install tensorflow to use CNN models.")
    return keras_load_model(model_path)

def preprocess_image_for_cnn(img):
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def predict_cnn(model, img_array):
    preds = model.predict(img_array)
    pred = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return pred, confidence
