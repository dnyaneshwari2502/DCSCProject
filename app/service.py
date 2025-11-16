# app/service.py

import pickle
import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# --- Load Artifacts ---
# This part runs only once when the application starts
MODEL_ARTIFACTS_PATH = 'app/models'
MODEL_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'multi_output_model.h5')
TOKENIZER_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'tokenizer.pkl')
CLICKBAIT_ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'clickbait_encoder.pkl')
EMOTION_ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'emotion_encoder.pkl')

# --- Check for model files and load them ---
if not all([os.path.exists(p) for p in [MODEL_PATH, TOKENIZER_PATH, CLICKBAIT_ENCODER_PATH, EMOTION_ENCODER_PATH]]):
    print("WARNING: Model artifacts not found. Prediction service will be in dummy mode.")
    model = None
    tokenizer = None
    clickbait_encoder = None
    emotion_encoder = None
else:
    print("Loading model and artifacts...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(CLICKBAIT_ENCODER_PATH, 'rb') as f:
        clickbait_encoder = pickle.load(f)
    with open(EMOTION_ENCODER_PATH, 'rb') as f:
        emotion_encoder = pickle.load(f)
    print("Model and artifacts loaded successfully.")

# --- NLTK and Preprocessing Setup ---
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))

try:
    # Test if 'punkt' is available
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')

try:
    # Test if 'wordnet' is available
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
LEMMATIZER = WordNetLemmatizer()
MAX_SEQUENCE_LENGTH = 100 # This must match the training parameter

# --- Prediction Logic ---
def preprocess_for_prediction(text: str) -> str:
    """Preprocesses a single raw string for prediction."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS and len(word) > 1]
    return ' '.join(cleaned_tokens)

def predict_clickbait_and_emotion(text: str) -> dict:
    """
    Takes raw text and returns a dictionary with clickbait and emotion predictions.
    """
    if model is None or tokenizer is None:
        # Dummy response if model isn't loaded
        return {
            "clickbait_prediction": "no-clickbait",
            "clickbait_score": 0.1,
            "emotion_prediction": "neutral_emotion",
            "emotion_score": 0.1,
            "model_status": "dummy"
        }

    # Preprocess, tokenize, and pad the input text
    cleaned_text = preprocess_for_prediction(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Make prediction
    predictions = model.predict(padded_sequence)
    clickbait_score = predictions[0][0][0]
    emotion_score = predictions[1][0][0]

    # Decode predictions
    clickbait_label_numeric = 1 if clickbait_score < 0.5 else 1 # Model predicts 1 for clickbait
    emotion_label_numeric = 1 if emotion_score > 0.5 else 0 # Model predicts 1 for sensational

    clickbait_prediction = clickbait_encoder.inverse_transform([clickbait_label_numeric])[0]
    emotion_prediction = emotion_encoder.inverse_transform([emotion_label_numeric])[0]

    return {
        "clickbait_prediction": clickbait_prediction,
        "clickbait_score": float(clickbait_score),
        "emotion_prediction": emotion_prediction,
        "emotion_score": float(emotion_score),
        "model_status": "loaded"
    }