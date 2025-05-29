import pandas as pd
import joblib

def load_model():
    """Load the trained ML model."""
    return joblib.load("best_model.pkl")

def predict(model, input_data):
    """Make prediction and flip it for correct interpretation."""
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    # Flip the prediction (0 becomes 1, 1 becomes 0) to correct the model output
    return 1 - prediction