import os

MODEL_PATH = os.path.join("models", "model.bin")  # placeholder

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Using dummy mode.")
        return None
    
    # later we'll load the actual model here
    print("Model file found. Loading real model...")
    return "loaded_model"

model = load_model()
