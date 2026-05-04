import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from training.data import load_raw, clean_city, merge
from training.features import add_features, add_targets, split, FEATURE_COLS, HORIZONS
from training.evaluate import compute_metrics
from utils.db import get_engine
from config import CITIES

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "air_quality_pm10"
N_TRIALS = 50


def build_dataset() -> dict[str, pd.DataFrame]:
    engine = get_engine()
    aq, weather = load_raw(engine)
    engine.dispose()

    city_dfs = {}
    for city in CITIES:
        cleaned = clean_city(aq, city)
        merged = merge(cleaned, weather)
        featured = add_features(merged)
        city_dfs[city] = add_targets(featured)
    return city_dfs


def tune_xgb(X_train, y_train, X_val, y_val) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = xgb.XGBRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return -np.sqrt(mean_squared_error(y_val, y_pred))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def train_city_horizon(df: pd.DataFrame, horizon: int):
    target_col = f"target_{horizon}d"
    train, val, test = split(df)

    X_train = train[FEATURE_COLS].values
    y_train = train[target_col].values
    X_val = val[FEATURE_COLS].values
    y_val = val[target_col].values
    X_test = test[FEATURE_COLS].values
    y_test = test[target_col].values

    best_params = tune_xgb(X_train, y_train, X_val, y_val)

    # retrain on train+val with tuned params before evaluating on test
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_fit, y_fit)

    metrics = compute_metrics(y_test, model.predict(X_test))
    return model, best_params, metrics


def promote_if_best(client, model_name: str, run_id: str, rmse: float):
    versions = client.search_model_versions(f"run_id='{run_id}' and name='{model_name}'")
    if not versions:
        return
    new_version = versions[0].version

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prod_rmse = client.get_run(prod_versions[0].run_id).data.metrics["rmse"]
        if rmse >= prod_rmse:
            print(f"keeping existing Production")
            return

    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"-> promoted version {new_version} to Production (RMSE {rmse:.2f})")


def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.tracking.MlflowClient()

    city_dfs = build_dataset()

    for city in CITIES:
        df = city_dfs[city]
        for h in HORIZONS:
            with mlflow.start_run(run_name=f"{city}_horizon_{h}d") as active_run:
                mlflow.set_tag("city", city)
                mlflow.set_tag("horizon", f"{h}d")

                model, best_params, metrics = train_city_horizon(df, h)

                mlflow.log_param("city", city)
                mlflow.log_param("horizon", h)
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)

                model_name = f"pm10_{city.lower()}_{h}d"
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )

                promote_if_best(client, model_name, active_run.info.run_id, metrics["rmse"])
                print(f"{city} +{h}d — RMSE: {metrics['rmse']:.2f}  MAE: {metrics['mae']:.2f}  R²: {metrics['r2']:.3f}")


if __name__ == "__main__":
    run()
