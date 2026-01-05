import os
import xgboost as xgb
import pandas as pd
import streamlit as st
import re

# Path to local models folder
MODELS_DIR = "models"


def sanitize_filename(name: str) -> str:
    import re
    # Replace spaces with underscores, removes special characters.
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_øæåØÆÅ]", "", name)
    return name


@st.cache_resource
def load_model(resort_name: str) -> xgb.XGBRegressor:
    #Load the XGBRegressor models from the local models folder
    filename = f"xgb_ordinal_model_{sanitize_filename(resort_name)}.json"
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
