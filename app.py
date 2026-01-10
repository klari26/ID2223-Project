import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

import hopsworks
from model_utils import load_model, predict
from dotenv import load_dotenv
import os
import re
import unicodedata

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

# Standarize location name
def sanitize_fg_name(resort_name: str):
    # Convert special characters to closest ASCII (ø → o, å → a, etc.)
    name_ascii = unicodedata.normalize('NFKD', resort_name).encode('ASCII', 'ignore').decode()

    # Lowercase and replace spaces with underscores
    name_ascii = name_ascii.lower()
    name_ascii = re.sub(r'\s+', '_', name_ascii)  # spaces → _
    name_ascii = re.sub(r'[^a-z0-9_]', '', name_ascii)  # remove other special chars
    name_ascii = re.sub(r'_+', '_', name_ascii)  # collapse multiple underscores
    return f"aq_predictions_{name_ascii}"[:63]  # max 63 chars

# Load resorts from CSV
@st.cache_data
def load_resorts():
    df = pd.read_csv("terrain_features.csv")
    return df

resorts_df = load_resorts()
resort_names = resorts_df['location'].tolist()

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

# Load all feature groups for resorts
@st.cache_data
def load_forecasts(resort_names):
    fg_dict = {}
    for resort in resort_names:
        fg_name = sanitize_fg_name(resort)
        try:
            fg = fs.get_feature_group(fg_name, version=1)
            fg_df = fg.read()  # get all 7-day predictions
            fg_dict[resort] = fg_df
        except Exception as e:
            st.warning(f"Could not load feature group for {resort}: {e}")
    return fg_dict

resort_names = resorts_df['location'].tolist()
forecasts = load_forecasts(resort_names)

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
tabs = st.tabs(["Map View", "Feature Details per Resort"])
tab_map, tab_details = tabs

# ==========================================================
# TAB 1 – Map
# ==========================================================
with tab_map:
    st.header("Avalanche Risk Map")
    st.info("High risk avalanche zones are marked in red on the map.")
    st.write("Select the day to visualize the prediction (1 = tomorrow, 7 = 7 days ahead).")

    # Select day
    selected_day = st.slider("Select Day", min_value=1, max_value=7, value=0, step=1)

    m = folium.Map(location=[64.5, 11], zoom_start=5)

    for _, row in resorts_df.iterrows():
        location = row["location"]
        if location not in forecasts:
            continue
        
        fg_df = forecasts[location]
        # Filter for the selected day
        day_df = fg_df[fg_df['days_before_forecast_day'] == selected_day]
        if day_df.empty:
            continue

        # One row per day
        pred_row = day_df.iloc[0]
        prediction = float(pred_row['warning_level_lag_1'])

        color = "red" if prediction >= 2 else "orange" if prediction >= 1 else "green"

        popup = f"""
        <b>{location}</b><br>
        Day: {selected_day}<br>
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
    st.header("7-Day Forecast per Resort")

    selected_resort = st.selectbox(
    "Select Resort",
    resort_names,
    key="tab_details_select_resort")

    if selected_resort in forecasts:
        resort_df = forecasts[selected_resort].copy()
        st.dataframe(resort_df)

        # Plot the 7-day predictions at the bottom
        import matplotlib.pyplot as plt

        # Aggregate risk per day (simple example: mean of warning_level_lags)
        resort_df['predicted_risk'] = resort_df[['warning_level_lag_1', 'warning_level_lag_2', 'warning_level_lag_3']].mean(axis=1)

        fig, ax = plt.subplots(figsize=(4,2))

        ax.plot(
            resort_df['days_before_forecast_day'],
            resort_df['predicted_risk'],
            marker='o',
            markersize=3,
            linewidth=1
        )

        ax.set_xticks(range(0, 7))
        ax.set_xlabel("Days Before Forecast Day", fontsize=6)
        ax.set_ylabel("Predicted Avalanche Risk", fontsize=6)
        ax.set_title(f"7-Day Avalanche Risk for {selected_resort}", fontsize=8)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True)
        fig.tight_layout()

        st.pyplot(fig, clear_figure=True, use_container_width=False)

    else:
        st.warning(f"No forecast available for {selected_resort}.")