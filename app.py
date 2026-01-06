import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

import hopsworks
from model_utils import load_model, predict
from dotenv import load_dotenv
import os

# Setting up Hopsworks
st.set_page_config(page_title="Avalanche Risk – Norway", layout="wide")

MODEL_FEATURES = [
    "warning_level_lag_1", 
    "warning_level_lag_2",
    "warning_level_lag_3",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "snow_load_steep",
    "wind_snow_transport",
    "rain_on_snow_risk",
    "temp_elev",
    "precip_slope_weighted",
]


# Load resorts from CSV
@st.cache_data
def load_resorts():
    df = pd.read_csv("terrain_features.csv")
    return df

resorts_df = load_resorts()

# Connect to Hopsworks
@st.cache_resource
def connect_hopsworks():
    load_dotenv()

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project="ID2223_Project",
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )
    fs = project.get_feature_store()
    return fs

fs = connect_hopsworks()

fv = fs.get_feature_view(
    name="avalanche_warning_fv_new_corrected_more_features_and_lags",
    version=3
)

# Get latest features per resort
@st.cache_data
def get_latest_features():
    #inference data
    batch_data = fv.get_batch_data()

    # Ensure datetime
    batch_data["date"] = pd.to_datetime(batch_data["date"])
    
    # Sort by location and date
    batch_data = batch_data.sort_values(["location", "date"])
    
    # Take the last row per location (latest)
    latest_df = batch_data.groupby("location", as_index=False).last()
    
    # Set location as index for easy lookup
    latest_df = latest_df.set_index("location")
    
    return latest_df

# Usage
latest_features = get_latest_features()



# UI
st.title("Norway Avalanche Forecast 🏔️🇳🇴")
st.write("Real-time avalanche risk predictions for Norwegian ski resorts")
tabs = st.tabs(["Map View", "Feature Details per Resort", "Scenario Simulation"])
tab_map, tab_details, tab_sim = tabs

# ==========================================================
# TAB 1 – Map
# ==========================================================
with tab_map:
    st.header("Avalanche Risk – Latest Forecast")
    st.info("High risk avalanche zones are marked in red on the map.")

    m = folium.Map(location=[64.5, 11], zoom_start=5)

    for _, row in resorts_df.iterrows():
        location = row["location"]

        if location not in latest_features.index:
            continue

        model = load_model(location)

        features = (
            latest_features
            .loc[location, MODEL_FEATURES]
            .to_frame()
            .T
        )

        prediction = predict(model, features)

        color = "red" if prediction >= 2 else "orange" if prediction >= 1 else "green"

        popup = f"""
        <b>{location}</b><br>
        Avalanche risk score: {prediction:.2f}<br>
        <small><i>For more details, review the Feature Details per Resort section</i></small>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)

    st_folium(m, width=1200, height=700)

# ==========================================================
# TAB 2 – Feature Details per Resort
# ==========================================================
with tab_details:
    st.header("Latest Weather Features")

    # Convert all numeric columns to float
    numeric_cols = [
        'warning_level_lag_1', 'warning_level_lag_2', 'warning_level_lag_3',
        'temperature_2m_mean', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
        'wind_speed_10m_max', 'wind_direction_10m_dominant', 'snow_load_steep',
        'wind_snow_transport', 'rain_on_snow_risk', 'temp_elev', 'precip_slope_weighted'
    ]

    # Rorce errors to NaN if anything invalid
    latest_features[numeric_cols] = latest_features[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Fill NaN with 0 
    latest_features[numeric_cols] = latest_features[numeric_cols].fillna(0)

    st.dataframe(latest_features)

# ==========================================================
# TAB 3 – Scenario Simulation
# ==========================================================

with tab_sim:
    st.header("Scenario Simulation: Modify Weather Features")
    st.write("Select a resort and adjust the weather features to see how the avalanche prediction changes.")

    # Select the resort
    resort_names = resorts_df['location'].tolist()
    selected_resort = st.selectbox("Select Resort", resort_names)

    # Load model for this resort
    model = load_model(selected_resort)

    # Features for sliders
    feature_cols = [
        'warning_level_lag_1', 'warning_level_lag_2', 'warning_level_lag_3', 
        'temperature_2m_mean', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
        'wind_speed_10m_max', 'wind_direction_10m_dominant', 'snow_load_steep',
        'wind_snow_transport', 'rain_on_snow_risk', 'temp_elev', 'precip_slope_weighted'
    ]

    # Create sliders for each feature with safe ranges
    user_features = {}
    for col in feature_cols:
        try:
            default_val = float(latest_features[col])
        except (KeyError, TypeError):
            default_val = 0.0  # default value if column missing or None

        # Ensure slider has valid min < max
        if default_val == 0.0:
            min_val = 0.0
            max_val = 1.0
        else:
            min_val = default_val * 0.5
            max_val = default_val * 1.5

        user_features[col] = st.slider(
            label=col.replace("_", " ").title(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1
        )

    # Convert user inputs to DataFrame
    user_features_df = pd.DataFrame([user_features])

    # Predict button
    if st.button("Predict Avalanche Risk"):
        prediction = predict(model, user_features_df)
        st.metric(
            label=f"Avalanche Risk Score for {selected_resort}",
            value=f"{prediction:.2f}"
        )

        # Interpretation level for prediction
        if prediction >= 2:
            st.warning("High avalanche risk!")
        elif prediction >= 1:
            st.info("Moderate avalanche risk.")
        else:
            st.success("Low avalanche risk.")