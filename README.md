# Air Quality Forecasting ML Pipeline

A machine learning pipeline that forecasts daily PM10 air quality for Berlin, London, Paris, and Amsterdam up to 7 days ahead. Built as a portfolio project to demonstrate end-to-end MLOps practices.

## Live demo

7-day forecasts and historical PM10 charts are available at:

**[https://ildogui22.github.io/air_quality_forecaster/](https://ildogui22.github.io/air_quality_forecaster/)**

## What it does

Every day, the pipeline fetches fresh air quality readings from the WAQI API and weather data from Open-Meteo, stores them in Postgres, and uses trained models to generate 7-day PM10 forecasts for each city.

## Models

For each city and forecast horizon (1 to 7 days), an **XGBoost** regressor is trained with hyperparameters tuned via Optuna (50 trials). Features include:

- **PM10 lags**: 1-day, 7-day, and 7-day rolling mean
- **Weather**: temperature, humidity, wind speed/gusts/direction, precipitation, snowfall, snow depth, cloud cover, surface pressure — on the target date, from the Open-Meteo forecast
- **Weather lags and rolling means**: 1-day and 3-day lags for key variables relative to the target date
- **Interaction terms**: wind × cloud, temp × humidity, precip × wind, snow × temp (with lags)
- **Seasonality**: month, day of week, and cyclic sin/cos encodings of day-of-year and day-of-week, computed for the target date

Training uses a strict chronological split to avoid data leakage: train ≤ 2022, validation = 2023 (used for Optuna tuning), test ≥ 2024. The final model is retrained on train + validation before evaluation. All experiments are tracked in MLflow on DagsHub, and the best model per city/horizon is automatically promoted to Production.

## Production stack

The entire stack runs on free-tier services:

- **Supabase** — hosted Postgres for air quality, weather, and prediction data
- **DagsHub** — hosted MLflow for experiment tracking and model registry
- **Render** — hosts the FastAPI backend
- **GitHub Actions** — runs the daily pipeline on a cron schedule and deploys the frontend to GitHub Pages

## Daily pipeline

GitHub Actions triggers `src/pipeline/pipeline.py` every day at 6am UTC. It checks the database for the last date present, backfills any missing weather data, updates actuals for past predictions where real PM10 has now arrived, and writes fresh 7-day forecasts.

## Extending to other cities

The city list lives in a single dictionary in [`src/config.py`](src/config.py):

```python
CITIES = {
    "Berlin":    {"lat": 52.52, "lon": 13.40, "station_id": 6132},
    "London":    {"lat": 51.51, "lon": -0.13, "station_id": 5724},
    ...
}
```

To add a city:
1. Find its WAQI station ID at [aqicn.org](https://aqicn.org) — search for the city and note the station ID from the URL
2. Download the historical CSV from aqicn.org, place it in `data/`, and add its path to `scripts/load_historical_data.py`
3. Add the entry to `CITIES` in `src/config.py` with coordinates and station ID
4. Bootstrap historical data (see below)
5. Retrain the models with `python -m training.train` from `src/`

## Bootstrapping historical data

**Air quality** — the WAQI API only returns the current day's reading, so historical data must be loaded from CSV files exported from [aqicn.org](https://aqicn.org/data-platform/token/):

```bash
python scripts/load_historical_data.py
```

**Weather** — the Open-Meteo archive API supports arbitrary date ranges, so historical weather is fetched automatically. The script reads the date range already present in `raw.air_quality` and backfills the matching weather:

```bash
python scripts/load_historical_weather.py
```
