import os
import xgboost as xgb
import pandas as pd
import streamlit as st
import re
import unicodedata

# Path to local models folder
MODELS_DIR = "models"


def sanitize_name(name):
    # Ignore accents
    name_ascii = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()
    # Replace anything not a-z, A-Z, 0-9, or _ with underscore
    name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', name_ascii)
    return name_clean

@st.cache_resource
def load_model(resort_name: str) -> xgb.XGBRegressor:
    #Load the XGBRegressor models from the local models folder
    filename = f"xgb_ordinal_model_more_features{sanitize_name(resort_name)}.json"
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create XGBRegressor
    model = xgb.XGBRegressor()
    model._estimator_type = "regressor" 
    model.load_model(model_path)

    return model


def predict(model: xgb.XGBRegressor, features_df: pd.DataFrame) -> float:
    """
    Predict avalanche risk for a given feature DataFrame.
    Returns a float value.
    """
    features_df = features_df.astype(float)
    return float(model.predict(features_df)[0])


def risk_label(value: float):
    """
    Convert numeric prediction to a risk label and map color for UI.
    """
    if value < 1.5:
        return "Low", "green"
    elif value < 2.5:
        return "Moderate", "orange"
    else:
        return "High", "red"