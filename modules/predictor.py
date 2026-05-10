from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config.settings import MODEL_DIR
from modules.database import log_model, query_df, replace_predictions
from modules.deep_predictor import (
    build_sequence_dataset,
    predict_with_cnn_bilstm,
    torch_available,
    train_cnn_bilstm,
)
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

TARGET_COLUMNS = ["target_start_count", "target_end_count"]


def load_grid_stats(engine=None) -> pd.DataFrame:
    """Read grid-hour statistics and merge weather/holiday features with safe defaults."""
    df = query_df("SELECT * FROM grid_hour_stats ORDER BY grid_id, stat_time", engine=engine)
    if df.empty:
        return df

    df = df.copy()
    df["stat_time"] = pd.to_datetime(df["stat_time"], errors="coerce")
    df = df.dropna(subset=["stat_time"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    env = load_environment_features(engine=engine)
    if not env.empty:
        keep = ["date", "temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]
        env = env[[col for col in keep if col in env.columns]].drop_duplicates(subset=["date"])
        env["date"] = pd.to_datetime(env["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = pd.merge(df, env, on="date", how="left")

    for col in ["start_count", "end_count", "temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Weather tables may be incomplete. Forward/backward fill by date, then use 0 as final fallback.
    df = df.sort_values(["date", "grid_id", "stat_time"])
    for col in ["temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]:
        df[col] = df[col].ffill().bfill().fillna(0)
    for col in ["start_count", "end_count"]:
        df[col] = df[col].fillna(0)
    return df


def build_features(stats: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling and historical features for grid-level virtual stations."""
    if stats.empty:
        return stats

    df = stats.sort_values(["grid_id", "stat_time"]).copy()
    df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce").fillna(pd.to_datetime(df["stat_time"]).dt.hour).astype(int)
    df["weekday"] = pd.to_numeric(df.get("weekday"), errors="coerce").fillna(pd.to_datetime(df["stat_time"]).dt.weekday).astype(int)
    df["is_weekend"] = pd.to_numeric(df.get("is_weekend"), errors="coerce").fillna((df["weekday"] >= 5).astype(int)).astype(int)

    for col in ["start_count", "end_count", "temperature_mean", "precipitation_sum", "wind_speed_max", "is_holiday"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = df.groupby("grid_id", group_keys=False)
    df["start_count_lag1"] = grouped["start_count"].shift(1).fillna(0)
    df["end_count_lag1"] = grouped["end_count"].shift(1).fillna(0)
    df["start_count_lag2"] = grouped["start_count"].shift(2).fillna(0)
    df["end_count_lag2"] = grouped["end_count"].shift(2).fillna(0)
    df["rolling_start_mean_3"] = grouped["start_count"].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0)
    df["rolling_end_mean_3"] = grouped["end_count"].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0)
    df["historical_start_mean"] = grouped["start_count"].transform(lambda s: s.expanding().mean()).fillna(0)
    df["historical_end_mean"] = grouped["end_count"].transform(lambda s: s.expanding().mean()).fillna(0)
    df["target_start_count"] = grouped["start_count"].shift(-1)
    df["target_end_count"] = grouped["end_count"].shift(-1)
    return df


def _environment_multiplier(env: dict) -> float:
    multiplier = 1.0
    if float(env.get("precipitation_sum", 0) or 0) > 0:
        multiplier -= 0.08
    if float(env.get("wind_speed_max", 0) or 0) >= 30:
        multiplier -= 0.08
    if float(env.get("temperature_mean", 0) or 0) >= 32:
        multiplier -= 0.06
    if int(env.get("is_holiday", 0) or 0) or int(env.get("is_weekend", 0) or 0):
        multiplier += 0.06
    return max(0.75, min(1.18, multiplier))


def add_demand_gap_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["predicted_start_count"] = pd.to_numeric(result["predicted_start_count"], errors="coerce").fillna(0).clip(lower=0).round(2)
    result["predicted_end_count"] = pd.to_numeric(result["predicted_end_count"], errors="coerce").fillna(0).clip(lower=0).round(2)
    result["demand_gap"] = (result["predicted_start_count"] - result["predicted_end_count"]).round(2)
    positive_gap = result.loc[result["demand_gap"] > 0, "demand_gap"]
    if positive_gap.empty:
        result["risk_level"] = "低"
    else:
        high_threshold = max(3.0, float(positive_gap.quantile(0.8)))
        medium_threshold = max(1.0, float(positive_gap.quantile(0.5)))
        result["risk_level"] = np.select(
            [result["demand_gap"] >= high_threshold, result["demand_gap"] >= medium_threshold],
            ["高", "中"],
            default="低",
        )
    result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


def _latest_feature_rows(features: pd.DataFrame, predict_time: datetime, engine=None) -> pd.DataFrame:
    latest = features.sort_values("stat_time").groupby("grid_id").tail(1).copy()
    latest["hour"] = predict_time.hour
    latest["weekday"] = predict_time.weekday()
    latest["is_weekend"] = 1 if predict_time.weekday() >= 5 else 0
    env = environment_for_date(predict_time.strftime("%Y-%m-%d"), engine=engine)
    latest["temperature_mean"] = env["temperature_mean"]
    latest["precipitation_sum"] = env["precipitation_sum"]
    latest["wind_speed_max"] = env["wind_speed_max"]
    latest["is_holiday"] = env["is_holiday"]
    return latest


def _prediction_frame(latest: pd.DataFrame, start_pred, end_pred, predict_time: datetime, model_name: str) -> pd.DataFrame:
    result = latest[["grid_id", "center_lng", "center_lat"]].copy()
    result["predict_time"] = predict_time.strftime("%Y-%m-%d %H:%M:%S")
    result["predicted_start_count"] = np.maximum(np.asarray(start_pred, dtype=float), 0).round(2)
    result["predicted_end_count"] = np.maximum(np.asarray(end_pred, dtype=float), 0).round(2)
    result["model_name"] = model_name
    return add_demand_gap_and_risk(result)


def predict_by_historical_trend(stats: pd.DataFrame, predict_time: datetime, engine=None) -> tuple[dict, pd.DataFrame]:
    """Fallback model: historical mean plus recent rolling trend correction."""
    features = build_features(stats)
    if features.empty:
        return {"model_name": "历史趋势模型", "sample_count": 0, "mae": None, "rmse": None, "r2": None, "model_path": "", "message": ""}, pd.DataFrame()

    latest = _latest_feature_rows(features, predict_time, engine=engine)
    env = environment_for_date(predict_time.strftime("%Y-%m-%d"), engine=engine)
    env_factor = _environment_multiplier(env)
    trend_start = latest["start_count_lag1"] - latest["start_count_lag2"]
    trend_end = latest["end_count_lag1"] - latest["end_count_lag2"]
    start_pred = (
        latest["historical_start_mean"] * 0.35
        + latest["rolling_start_mean_3"] * 0.40
        + latest["start_count"] * 0.20
        + trend_start * 0.25
    ) * env_factor
    end_pred = (
        latest["historical_end_mean"] * 0.35
        + latest["rolling_end_mean_3"] * 0.40
        + latest["end_count"] * 0.20
        + trend_end * 0.25
    ) * env_factor

    predictions = _prediction_frame(latest, start_pred, end_pred, predict_time, "历史均值 + 最近趋势修正")
    train_df = features.dropna(subset=TARGET_COLUMNS)
    return {
        "model_name": "历史均值 + 最近趋势修正",
        "sample_count": int(len(train_df)),
        "mae": None,
        "rmse": None,
        "r2": None,
        "model_path": "",
        "message": "样本较少，系统采用历史均值与最近趋势修正保证预测流程可运行。",
    }, predictions


def predict_by_gradient_boosting(stats: pd.DataFrame, predict_time: datetime, engine=None) -> tuple[dict, pd.DataFrame]:
    """Train lightweight scikit-learn models for start/end demand separately."""
    start_time = datetime.now()
    features = build_features(stats)
    train_df = features.dropna(subset=TARGET_COLUMNS).copy()
    if len(train_df) < 20:
        return predict_by_historical_trend(stats, predict_time, engine=engine)

    X = train_df[FEATURE_COLUMNS].fillna(0)
    y_start = train_df["target_start_count"].astype(float).clip(lower=0)
    y_end = train_df["target_end_count"].astype(float).clip(lower=0)
    test_size = 0.25 if len(train_df) >= 80 else 0.3
    X_train, X_test, y_start_train, y_start_test, y_end_train, y_end_test = train_test_split(
        X, y_start, y_end, test_size=test_size, random_state=42
    )
    start_model = GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=3)
    end_model = GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=3)
    start_model.fit(X_train, y_start_train)
    end_model.fit(X_train, y_end_train)

    pred_start_test = np.maximum(start_model.predict(X_test), 0)
    pred_end_test = np.maximum(end_model.predict(X_test), 0)
    true = np.concatenate([y_start_test.to_numpy(), y_end_test.to_numpy()])
    pred = np.concatenate([pred_start_test, pred_end_test])
    mae = float(mean_absolute_error(true, pred))
    rmse = float(mean_squared_error(true, pred) ** 0.5)
    r2 = float(r2_score(true, pred)) if len(true) > 1 else None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(MODEL_DIR) / f"gb_grid_demand_{start_time.strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump({"start_model": start_model, "end_model": end_model, "features": FEATURE_COLUMNS}, model_path)

    latest = _latest_feature_rows(features, predict_time, engine=engine)
    X_latest = latest[FEATURE_COLUMNS].fillna(0)
    predictions = _prediction_frame(
        latest,
        start_model.predict(X_latest),
        end_model.predict(X_latest),
        predict_time,
        "GradientBoostingRegressor",
    )
    return {
        "model_name": "GradientBoostingRegressor",
        "sample_count": int(len(train_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path,
        "message": "系统采用轻量梯度提升模型分别预测借车需求与还车供给。",
    }, predictions


def _latest_sequences(features: pd.DataFrame, feature_columns: list[str], window: int = 6) -> tuple[np.ndarray, pd.DataFrame]:
    seqs = []
    meta = []
    df = features.sort_values(["grid_id", "stat_time"]).copy()
    for grid_id, group in df.groupby("grid_id"):
        group = group.sort_values("stat_time")
        if len(group) < window:
            continue
        seqs.append(group[feature_columns].fillna(0).astype(float).tail(window).values)
        last = group.iloc[-1]
        meta.append({"grid_id": grid_id, "center_lng": last["center_lng"], "center_lat": last["center_lat"]})
    if not seqs:
        return np.empty((0, window, len(feature_columns))), pd.DataFrame()
    return np.asarray(seqs, dtype=np.float32), pd.DataFrame(meta)


def predict_by_cnn_bilstm(stats: pd.DataFrame, predict_time: datetime, engine=None) -> tuple[dict, pd.DataFrame]:
    """Try CNN-BiLSTM; callers should fall back when this returns empty predictions."""
    if not torch_available():
        return {"success": False, "message": "当前环境未安装 PyTorch，系统已自动使用轻量预测模型。"}, pd.DataFrame()

    features = build_features(stats)
    train_result = train_cnn_bilstm(features, FEATURE_COLUMNS, window=6, epochs=35)
    if not train_result.get("success"):
        return train_result, pd.DataFrame()

    latest_sequences, meta = _latest_sequences(features, FEATURE_COLUMNS, window=6)
    if latest_sequences.size == 0 or meta.empty:
        return {"success": False, "message": "可预测序列不足，跳过 CNN-BiLSTM。"}, pd.DataFrame()

    pred = predict_with_cnn_bilstm(latest_sequences)
    meta["predict_time"] = predict_time.strftime("%Y-%m-%d %H:%M:%S")
    meta["predicted_start_count"] = pred[:, 0].round(2)
    meta["predicted_end_count"] = pred[:, 1].round(2)
    meta["model_name"] = "CNN-BiLSTM"
    predictions = add_demand_gap_and_risk(meta)
    # Some low-frequency grids may not have a full 6-hour sequence; keep full-area output by filling them with the trend fallback.
    _, fallback = predict_by_historical_trend(stats, predict_time, engine=engine)
    if not fallback.empty:
        missing = fallback[~fallback["grid_id"].astype(str).isin(predictions["grid_id"].astype(str))].copy()
        if not missing.empty:
            missing["model_name"] = "CNN-BiLSTM补充趋势"
            predictions = pd.concat([predictions, missing], ignore_index=True)
            predictions = add_demand_gap_and_risk(predictions)
    metrics = {
        "model_name": "CNN-BiLSTM",
        "sample_count": int(train_result.get("sequence_count", 0)),
        "mae": train_result.get("mae"),
        "rmse": train_result.get("rmse"),
        "r2": train_result.get("r2"),
        "model_path": train_result.get("model_path", ""),
        "message": "系统基于过去 6 小时网格序列训练 CNN-BiLSTM，并预测下一小时借还需求。",
    }
    return metrics, predictions


def smart_predict(stats: pd.DataFrame, predict_time: datetime, engine=None) -> tuple[dict, pd.DataFrame]:
    features = build_features(stats)
    sample_count = int(features.dropna(subset=TARGET_COLUMNS).shape[0]) if not features.empty else 0

    if sample_count < 50:
        return predict_by_historical_trend(stats, predict_time, engine=engine)

    if sample_count >= 200:
        cnn_metrics, cnn_predictions = predict_by_cnn_bilstm(stats, predict_time, engine=engine)
        if not cnn_predictions.empty:
            return cnn_metrics, cnn_predictions
        try:
            gb_metrics, gb_predictions = predict_by_gradient_boosting(stats, predict_time, engine=engine)
            cnn_message = cnn_metrics.get("message", "CNN-BiLSTM 未启用。")
            if "轻量预测模型" in cnn_message:
                gb_metrics["message"] = cnn_message
            else:
                gb_metrics["message"] = f"{cnn_message} 当前已回退到轻量预测模型。"
            return gb_metrics, gb_predictions
        except Exception as exc:
            metrics, predictions = predict_by_historical_trend(stats, predict_time, engine=engine)
            metrics["message"] = f"CNN-BiLSTM 未启用且轻量模型训练失败，已回退到历史趋势模型：{exc}"
            return metrics, predictions

    try:
        return predict_by_gradient_boosting(stats, predict_time, engine=engine)
    except Exception as exc:
        metrics, predictions = predict_by_historical_trend(stats, predict_time, engine=engine)
        metrics["message"] = f"轻量模型训练失败，已回退到历史趋势模型：{exc}"
        return metrics, predictions


def train_and_save_predictions(engine=None, predict_time: datetime | None = None) -> tuple[dict, pd.DataFrame]:
    predict_time = predict_time or datetime.now() + timedelta(hours=1)
    ensure_environment_for_date(predict_time.strftime("%Y-%m-%d"), engine=engine)
    stats = load_grid_stats(engine=engine)
    if stats.empty:
        return {"model_name": "", "sample_count": 0, "mae": None, "rmse": None, "r2": None, "model_path": "", "message": "暂无网格小时统计数据。"}, pd.DataFrame()

    train_start = datetime.now()
    metrics, predictions = smart_predict(stats, predict_time, engine=engine)
    train_end = datetime.now()
    replace_predictions(predictions, engine=engine)
    log_model(
        {
            "model_name": metrics.get("model_name", ""),
            "train_start_time": train_start.strftime("%Y-%m-%d %H:%M:%S"),
            "train_end_time": train_end.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": int(metrics.get("sample_count", 0) or 0),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2"),
            "model_path": metrics.get("model_path", ""),
            "created_at": train_end.strftime("%Y-%m-%d %H:%M:%S"),
        },
        engine=engine,
    )
    return metrics, predictions


def train_and_predict(engine=None, predict_time: datetime | None = None) -> tuple[dict, pd.DataFrame]:
    """Compatibility alias used by Streamlit pages."""
    return train_and_save_predictions(engine=engine, predict_time=predict_time)


# Backward-compatible names kept for pages or notebooks that imported old helpers.
historical_mean_predict = lambda stats, predict_time: predict_by_historical_trend(stats, predict_time)[1]
train_random_forest = lambda stats, engine=None: predict_by_gradient_boosting(stats, datetime.now() + timedelta(hours=1), engine=engine)[0]
predict_next_hour = lambda stats, predict_time=None, model_path=None, engine=None: predict_by_historical_trend(stats, predict_time or datetime.now() + timedelta(hours=1), engine=engine)[1]


def top_k_hotspot_hit_rate(engine=None, k: int = 10) -> float | None:
    """Compare predicted Top-K shortage grids with latest observed Top-K start-demand grids."""
    pred = query_df(
        """
        SELECT grid_id, COALESCE(demand_gap, predicted_start_count - predicted_end_count) AS demand_gap
        FROM prediction_results
        ORDER BY demand_gap DESC
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
        SELECT grid_id, start_count
        FROM grid_hour_stats
        WHERE stat_time = :stat_time
        ORDER BY start_count DESC
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
