# Air Quality Forecasting ML Pipeline

A machine learning pipeline that forecasts daily PM10 air quality for Berlin, London, Paris, and Amsterdam up to 7 days ahead. Built as a portfolio project to demonstrate end-to-end MLOps practices.

## What it does

Every day, the pipeline fetches fresh air quality readings from the WAQI API and weather data from Open-Meteo, stores them in Postgres, and uses trained models to generate 7-day PM10 forecasts for each city. Predictions are served through a REST API.

## Models

For each city and forecast horizon (1 to 7 days), an **XGBoost** regressor is trained with hyperparameters tuned via Optuna (50 trials). Features include:

- **PM10 lags**: 1-day, 7-day, and 7-day rolling mean
- **Weather**: temperature, humidity, wind speed/gusts/direction, precipitation, snowfall, snow depth, cloud cover, surface pressure
- **Weather lags and rolling means**: 1-day and 3-day lags for key variables
- **Interaction terms**: wind × cloud, temp × humidity, precip × wind, snow × temp (with lags)
- **Seasonality**: month, day of week, and cyclic sin/cos encodings of day-of-year and day-of-week

Training uses a strict chronological split to avoid data leakage: train ≤ 2022, validation = 2023 (used for Optuna tuning), test ≥ 2024. The final model is retrained on train + validation before evaluation. All experiments are tracked in MLflow on DagsHub, and the best model per city/horizon is automatically promoted to Production.

## API
### Requests may take a while as Render sleeps after long downtime

Base URL: `https://air-quality-forecaster-lrz1.onrender.com`

**7-day forecast for a city:**
```bash
curl https://air-quality-forecaster-lrz1.onrender.com/forecast/berlin
```

**Historical PM10 readings:**
```bash
curl https://air-quality-forecaster-lrz1.onrender.com/history/london?days=14
```

Supported cities: `berlin`, `london`, `paris`, `amsterdam`

## Production stack

The entire stack runs on free-tier services:

- **Supabase** — hosted Postgres for air quality, weather, and prediction data
- **Dagshub** — hosted MLflow for experiment tracking and model registry
- **Render** — hosts the FastAPI backend
- **GitHub Actions** — runs the daily pipeline on a cron schedule

## Daily pipeline

GitHub Actions triggers `src/pipeline/pipeline.py` every day at 6am UTC. It checks the database for the last date present, backfills any missing weather data, updates actuals for past predictions where real PM10 has now arrived, and writes fresh 7-day forecasts.
