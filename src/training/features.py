import numpy as np
import pandas as pd

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "precipitation", "snowfall", "snow_depth",
    "cloud_cover", "surface_pressure", "weather_code"
]

WEATHER_LAG_COLS = [
    "wind_speed_lag_1d", "wind_speed_lag_3d", "wind_speed_rolling_3d",
    "temperature_lag_1d", "temperature_lag_3d",
    "precipitation_rolling_3d",
    "humidity_lag_1d",
    "wind_x_cloud_lag_1d",
    "temp_x_humidity_lag_1d",
    "precip_x_wind_lag_1d",
    "precip_x_wind_lag_3d",
]

FEATURE_COLS = (
    ["pm10_lag_1d", "pm10_lag_7d", "pm10_rolling_7d"]
    + WEATHER_COLS
    + WEATHER_LAG_COLS
    + [
        "wind_x_cloud", "temp_x_humidity", "precip_x_wind", "snow_x_temp",
        "month", "year", "day_of_week",
        "day_of_year_sin", "day_of_year_cos",
        "day_of_week_sin", "day_of_week_cos",
    ]
)

HORIZONS = list(range(1, 8))

TRAIN_END = "2022-12-31"
VALIDATION_END = "2023-12-31"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PM10 lags
    df["pm10_lag_1d"] = df.groupby("city")["pm10"].shift(1)
    df["pm10_lag_7d"] = df.groupby("city")["pm10"].shift(7)
    df["pm10_rolling_7d"] = df.groupby("city")["pm10"].transform(lambda x: x.shift(1).rolling(7).mean())

    # Weather lags (relative to each row's date
    df["wind_speed_lag_1d"] = df.groupby("city")["wind_speed_10m"].shift(1)
    df["wind_speed_lag_3d"] = df.groupby("city")["wind_speed_10m"].shift(3)
    df["wind_speed_rolling_3d"] = df.groupby("city")["wind_speed_10m"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["temperature_lag_1d"] = df.groupby("city")["temperature_2m"].shift(1)
    df["temperature_lag_3d"] = df.groupby("city")["temperature_2m"].shift(3)
    df["precipitation_rolling_3d"] = df.groupby("city")["precipitation"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["humidity_lag_1d"] = df.groupby("city")["relative_humidity_2m"].shift(1)

    # Seasonality
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Interactions
    df["wind_x_cloud"] = df["wind_speed_10m"] * df["cloud_cover"]
    df["temp_x_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["precip_x_wind"] = df["precipitation"] * df["wind_speed_10m"]
    df["snow_x_temp"] = df["snow_depth"] * df["temperature_2m"]

    # Interaction lags
    df["wind_x_cloud_lag_1d"] = df.groupby("city")["wind_x_cloud"].shift(1)
    df["temp_x_humidity_lag_1d"] = df.groupby("city")["temp_x_humidity"].shift(1)
    df["precip_x_wind_lag_1d"] = df.groupby("city")["precip_x_wind"].shift(1)
    df["precip_x_wind_lag_3d"] = df.groupby("city")["precip_x_wind"].shift(3)

    return df


def add_targets(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    if horizons is None:
        horizons = HORIZONS
    df = df.copy()
    for h in horizons:
        df[f"target_{h}d"] = df.groupby("city")["pm10"].shift(-h)
    return df.dropna(subset=[f"target_{h}d" for h in horizons])


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split into train / val / test sets.
    Train: up to TRAIN_END (≤ 2022)
    Validation: TRAIN_END – VALIDATION_END (2023) — hyperparameter tuning
    Test: after VALIDATION_END (2024+) — final evaluation
    """
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VALIDATION_END)]
    test = df[df["date"] > VALIDATION_END]
    return train, val, test

