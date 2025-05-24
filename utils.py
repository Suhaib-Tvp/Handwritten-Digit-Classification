import numpy as np
from PIL import Image, ImageOps
import joblib
from tensorflow.keras.models import load_model as keras_load_model

def load_logistic_model(model_path='logistic_model.pkl', scaler_path='scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_cnn_model(model_path='cnn_model.h5'):
    return keras_load_model(model_path)

def preprocess_image_for_logistic(img, scaler):
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, -1)
    img_array = scaler.transform(img_array)
    return img_array

def preprocess_image_for_cnn(img):
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def predict_logistic(model, img_array):
    pred = model.predict(img_array)[0]
    confidence = np.max(model.predict_proba(img_array))
    return pred, confidence

def predict_cnn(model, img_array):
    preds = model.predict(img_array)
    pred = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return pred, confidence
