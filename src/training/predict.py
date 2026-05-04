import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow.pyfunc
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import text

from training.data import load_raw, clean_city
from training.features import add_features, FEATURE_COLS, HORIZONS
from utils.db import get_engine, ensure_predictions_table
from config import CITIES

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_production_model(city: str, horizon: int):
    model_name = f"pm10_{city.lower()}_{horizon}d"
    return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")


def load_weather_forecast(engine, forecast_date: date) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM raw.weather_forecast WHERE forecast_date = :d"),
            {"d": forecast_date},
        ).mappings().all()
    df = pd.DataFrame(rows)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


def build_horizon_features(
    aq: pd.DataFrame,
    weather: pd.DataFrame,
    weather_fc: pd.DataFrame,
    city: str,
    forecast_date: date,
) -> pd.DataFrame:
    """
    Returns one feature row per horizon (1–7) for a city.

    Historical weather provides lag features; forecast weather provides the
    actual weather values on each target date. PM10 lags are always computed
    from the last known historical values (future PM10 is unknown).
    """
    # Reshape forecast rows to match historical weather schema
    fc_city = weather_fc[weather_fc["city"] == city].copy()
    fc_city = fc_city.rename(columns={"target_date": "date"}).drop(columns=["forecast_date"])

    # Combine historical + forecast weather, historical wins on overlap
    weather_hist = weather[weather["city"] == city].copy()
    combined_weather = (
        pd.concat([weather_hist, fc_city], ignore_index=True)
        .drop_duplicates(subset=["city", "date"], keep="first")
        .sort_values(["city", "date"])
        .reset_index(drop=True)
    )

    # Attach historical PM10 — NaN for future dates, which is correct:
    # lag features shift back into known history for all 7 horizons
    aq_city = clean_city(aq, city)[["date", "city", "pm10"]]
    combined = combined_weather.merge(aq_city, on=["city", "date"], how="left")

    featured = add_features(combined)

    # PM10 lags must reflect today's knowledge, not the target date's.
    # For h>1, pm10_lag_1d on the target-date row would be NaN (future unknown),
    # so we anchor all PM10 lag values to the latest historical row.
    today = pd.Timestamp(forecast_date)
    today_row = featured[featured["date"] <= today].sort_values("date").iloc[-1]
    pm10_lags = {col: today_row[col] for col in ["pm10_lag_1d", "pm10_lag_7d", "pm10_rolling_7d"]}

    # Extract one row per target date
    target_dates = [today + pd.Timedelta(days=h) for h in HORIZONS]
    rows = featured[featured["date"].isin(target_dates)].copy()
    rows["horizon"] = (rows["date"] - today).dt.days

    for col, val in pm10_lags.items():
        rows[col] = val

    return rows


def run_inference(forecast_date: date = None):
    if forecast_date is None:
        forecast_date = date.today()

    mlflow.set_tracking_uri(MLFLOW_URI)

    engine = get_engine()
    ensure_predictions_table(engine)
    aq, weather = load_raw(engine)
    weather_fc = load_weather_forecast(engine, forecast_date)

    rows = []
    for city in CITIES:
        fc_features = build_horizon_features(aq, weather, weather_fc, city, forecast_date)

        for h in HORIZONS:
            horizon_row = fc_features[fc_features["horizon"] == h]
            if horizon_row.empty:
                print(f"{city} +{h}d: no forecast weather, skipping")
                continue

            model = load_production_model(city, h)
            pred = model.predict(horizon_row[FEATURE_COLS])[0]

            rows.append({
                "city": city,
                "forecast_date": forecast_date,
                "target_date": forecast_date + pd.Timedelta(days=h),
                "horizon": h,
                "predicted": float(pred),
                "actual": None,
            })

    df = pd.DataFrame(rows)
    with engine.begin() as conn:
        for _, row in df.iterrows():
            # updates row and overwrite if data already exists
            conn.execute(text("""
                INSERT INTO raw.predictions (city, forecast_date, target_date, horizon, predicted, actual)
                VALUES (:city, :forecast_date, :target_date, :horizon, :predicted, :actual)
                ON CONFLICT (city, forecast_date, horizon) DO UPDATE SET predicted = EXCLUDED.predicted
            """), row.to_dict())

    engine.dispose()
    print(f"Wrote {len(rows)} predictions for {forecast_date}")


if __name__ == "__main__":
    run_inference()
