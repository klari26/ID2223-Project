import os
import datetime
import time
import requests
import pandas as pd
import openmeteo_requests
import requests_cache
import json
from retry_requests import retry
from datetime import datetime, timedelta

def get_historical_weather(location, start_date, end_date, longitude, latitude):

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # https://www.ortovox.com/en/safety-academy-lab-snow/01-avalanche-basics/avalanche-factors?srsltid=AfmBOoqL1e3qn7kWWD_iXVO97V9wTVBficdosC_jTjJSMCfKDkU7MTjf
    # Critical amount of new snow + wind + temperature + precipitation
    daily_variables = [
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
    ]

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_variables
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    # Build date range
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }

    # Extract each variable dynamically based on index
    for idx, var_name in enumerate(daily_variables):
        daily_data[var_name] = daily.Variables(idx).ValuesAsNumpy()

    # Convert to DataFrame
    df = pd.DataFrame(daily_data).dropna()
    df['location'] = location
    return df

def get_warning_data(start_date, end_date, lat, lon, lang=2):
    url = (
        f"https://api01.nve.no/hydrology/forecast/avalanche/v6.3.0/api/"
        f"AvalancheWarningByCoordinates/Simple/"
        f"{lat}/{lon}/{lang}/{start_date}/{end_date}"
    )

    headers = {"User-Agent": "avalanche-research-app"}

    print(url)

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"API error {start_date}–{end_date}: {e}")
        return []

def date_chunks(start_date, end_date, chunk_days=60):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while start < end:
        chunk_end = min(start + timedelta(days=chunk_days), end)
        yield start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        start = chunk_end + timedelta(days=1)