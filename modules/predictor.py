from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config.settings import MODEL_DIR
from modules.database import log_model, query_df, replace_predictions
from modules.environment import environment_for_date, ensure_environment_for_date, load_environment_features
from modules.grid import get_grid_center


FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "is_weekend",
    "start_count",
    "end_count",
    "start_count_lag1",
    "end_count_lag1",
    "start_count_lag2",
    "end_count_lag2",
    "historical_start_mean",
    "historical_end_mean",
    "rolling_start_mean_3",
    "rolling_end_mean_3",
    "temperature_mean",
    "precipitation_sum",
    "wind_speed_max",
    "is_holiday",
]


def load_grid_stats(engine=None) -> pd.DataFrame:
    df = query_df("SELECT * FROM grid_hour_stats ORDER BY grid_id, stat_time", engine=engine)
    if df.empty:
        return df
    df["stat_time"] = pd.to_datetime(df["stat_time"])
    env = load_environment_features(engine=engine)
    if not env.empty:
        keep = ["date", "temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]
        env = env[[col for col in keep if col in env.columns]].drop_duplicates(subset=["date"])
        df = pd.merge(df, env, on="date", how="left")
    for col in ["temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def build_features(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return stats
    df = stats.sort_values(["grid_id", "stat_time"]).copy()
    for col in ["start_count", "end_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    grouped = df.groupby("grid_id", group_keys=False)
    df["start_count_lag1"] = grouped["start_count"].shift(1).fillna(0)
    df["end_count_lag1"] = grouped["end_count"].shift(1).fillna(0)
    df["start_count_lag2"] = grouped["start_count"].shift(2).fillna(0)
    df["end_count_lag2"] = grouped["end_count"].shift(2).fillna(0)
    df["historical_start_mean"] = grouped["start_count"].transform(lambda s: s.expanding().mean()).fillna(0)
    df["historical_end_mean"] = grouped["end_count"].transform(lambda s: s.expanding().mean()).fillna(0)
    df["rolling_start_mean_3"] = grouped["start_count"].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0)
    df["rolling_end_mean_3"] = grouped["end_count"].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0)
    df["target_end_count"] = grouped["end_count"].shift(-1)
    df["target_start_count"] = grouped["start_count"].shift(-1)
    return df


def historical_mean_predict(stats: pd.DataFrame, predict_time: datetime) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    hour = predict_time.hour
    result_rows = []
    global_hour = stats.groupby("hour")[["end_count", "start_count"]].mean()
    overall = stats[["end_count", "start_count"]].mean()
    for grid_id, g in stats.groupby("grid_id"):
        same_hour = g[g["hour"] == hour]
        if not same_hour.empty:
            end_pred = same_hour["end_count"].mean()
            start_pred = same_hour["start_count"].mean()
        elif hour in global_hour.index:
            end_pred = global_hour.loc[hour, "end_count"]
            start_pred = global_hour.loc[hour, "start_count"]
        else:
            end_pred = overall["end_count"]
            start_pred = overall["start_count"]
        center_lng, center_lat = get_grid_center(grid_id)
        result_rows.append(
            {
                "grid_id": grid_id,
                "predict_time": predict_time.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_end_count": round(float(end_pred), 2),
                "predicted_start_count": round(float(start_pred), 2),
                "model_name": "历史均值模型",
                "center_lng": center_lng,
                "center_lat": center_lat,
            }
        )
    return add_risk_level(pd.DataFrame(result_rows))


def add_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    q80 = df["predicted_end_count"].quantile(0.8)
    q50 = df["predicted_end_count"].quantile(0.5)
    df["risk_level"] = np.select(
        [df["predicted_end_count"] >= q80, df["predicted_end_count"] >= q50],
        ["高", "中"],
        default="低",
    )
    df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


def train_random_forest(stats: pd.DataFrame, engine=None) -> dict:
    start = datetime.now()
    features = build_features(stats)
    train_df = features.dropna(subset=["target_end_count", "target_start_count"]).copy()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if len(train_df) < 10:
        model_path = ""
        metrics = {"mae": None, "rmse": None, "r2": None}
        model_name = "历史均值模型"
    else:
        X = train_df[FEATURE_COLUMNS].fillna(0)
        y_end = train_df["target_end_count"].astype(float)
        y_start = train_df["target_start_count"].astype(float)
        test_size = 0.25 if len(train_df) >= 20 else 0.3
        X_train, X_test, y_end_train, y_end_test, y_start_train, y_start_test = train_test_split(
            X, y_end, y_start, test_size=test_size, random_state=42
        )
        end_model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=1)
        start_model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=1)
        end_model.fit(X_train, y_end_train)
        start_model.fit(X_train, y_start_train)
        pred = end_model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_end_test, pred)),
            "rmse": float(mean_squared_error(y_end_test, pred) ** 0.5),
            "r2": float(r2_score(y_end_test, pred)) if len(y_end_test) > 1 else None,
        }
        model_name = "RandomForestRegressor"
        model_path = str(Path(MODEL_DIR) / f"rf_grid_model_{start.strftime('%Y%m%d_%H%M%S')}.joblib")
        joblib.dump({"end_model": end_model, "start_model": start_model, "features": FEATURE_COLUMNS}, model_path)

    end = datetime.now()
    log_model(
        {
            "model_name": model_name,
            "train_start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "train_end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": int(len(train_df)),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "model_path": model_path,
            "created_at": end.strftime("%Y-%m-%d %H:%M:%S"),
        },
        engine=engine,
    )
    return {"model_name": model_name, "model_path": model_path, "sample_count": len(train_df), **metrics}


def predict_next_hour(stats: pd.DataFrame, predict_time: datetime | None = None, model_path: str | None = None, engine=None) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    predict_time = predict_time or (pd.to_datetime(stats["stat_time"]).max().to_pydatetime() + timedelta(hours=1))
    if not model_path or not Path(model_path).exists():
        return historical_mean_predict(stats, predict_time)

    payload = joblib.load(model_path)
    features = build_features(stats)
    latest = features.sort_values("stat_time").groupby("grid_id").tail(1).copy()
    latest["hour"] = predict_time.hour
    latest["weekday"] = predict_time.weekday()
    latest["is_weekend"] = 1 if predict_time.weekday() >= 5 else 0
    env = environment_for_date(predict_time.strftime("%Y-%m-%d"), engine=engine)
    latest["temperature_mean"] = env["temperature_mean"]
    latest["precipitation_sum"] = env["precipitation_sum"]
    latest["wind_speed_max"] = env["wind_speed_max"]
    latest["is_holiday"] = env["is_holiday"]
    X = latest[FEATURE_COLUMNS].fillna(0)
    latest["predicted_end_count"] = np.maximum(payload["end_model"].predict(X), 0).round(2)
    latest["predicted_start_count"] = np.maximum(payload["start_model"].predict(X), 0).round(2)
    latest["predict_time"] = predict_time.strftime("%Y-%m-%d %H:%M:%S")
    latest["model_name"] = "RandomForestRegressor"
    latest["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = latest[
        [
            "grid_id",
            "predict_time",
            "predicted_end_count",
            "predicted_start_count",
            "model_name",
            "center_lng",
            "center_lat",
            "created_at",
        ]
    ].copy()
    return add_risk_level(result)


def train_and_save_predictions(engine=None, predict_time: datetime | None = None) -> tuple[dict, pd.DataFrame]:
    if predict_time is not None:
        ensure_environment_for_date(predict_time.strftime("%Y-%m-%d"), engine=engine)
    stats = load_grid_stats(engine=engine)
    if stats.empty:
        return {"model_name": "", "sample_count": 0, "mae": None, "rmse": None, "r2": None, "model_path": ""}, pd.DataFrame()
    metrics = train_random_forest(stats, engine=engine)
    predictions = predict_next_hour(stats, predict_time=predict_time, model_path=metrics.get("model_path"), engine=engine)
    replace_predictions(predictions, engine=engine)
    return metrics, predictions


def top_k_hotspot_hit_rate(engine=None, k: int = 10) -> float | None:
    """Compare predicted Top-K pressure grids with latest observed Top-K grids."""
    pred = query_df(
        """
        SELECT grid_id, predicted_end_count
        FROM prediction_results
        ORDER BY predicted_end_count DESC
        LIMIT :k
        """,
        params={"k": k},
        engine=engine,
    )
    latest_time = query_df("SELECT MAX(stat_time) AS stat_time FROM grid_hour_stats", engine=engine)
    if pred.empty or latest_time.empty or not latest_time["stat_time"].iloc[0]:
        return None
    actual = query_df(
        """
        SELECT grid_id, end_count
        FROM grid_hour_stats
        WHERE stat_time = :stat_time
        ORDER BY end_count DESC
        LIMIT :k
        """,
        params={"stat_time": latest_time["stat_time"].iloc[0], "k": k},
        engine=engine,
    )
    if actual.empty:
        return None
    pred_set = set(pred["grid_id"].astype(str))
    actual_set = set(actual["grid_id"].astype(str))
    if not pred_set or not actual_set:
        return None
    return len(pred_set & actual_set) / min(k, len(actual_set))
