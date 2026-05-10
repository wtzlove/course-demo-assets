from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import pandas as pd

from modules.database import query_df, replace_dispatch_plans


def haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    r = 6371.0
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(a))


def generate_dispatch_plan(predictions: pd.DataFrame, threshold: float = 1.0, max_tasks: int = 30) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["surplus_score"] = df["predicted_end_count"] - df["predicted_start_count"]
    surplus = df[df["surplus_score"] > threshold].sort_values("surplus_score", ascending=False).copy()
    shortage = df[df["surplus_score"] < -threshold].sort_values("surplus_score").copy()
    plans = []
    shortage_state = {r.grid_id: abs(float(r.surplus_score)) for r in shortage.itertuples()}

    for s in surplus.itertuples():
        available = int(round(float(s.surplus_score)))
        if available <= 0:
            continue
        candidates = []
        for t in shortage.itertuples():
            need = shortage_state.get(t.grid_id, 0)
            if need <= 0:
                continue
            distance = haversine_km(s.center_lng, s.center_lat, t.center_lng, t.center_lat)
            candidates.append((distance, t, need))
        for distance, t, need in sorted(candidates, key=lambda x: x[0]):
            if available <= 0 or len(plans) >= max_tasks:
                break
            bikes = int(min(available, round(need)))
            if bikes <= 0:
                continue
            priority = "高" if bikes >= 5 or distance <= 1.0 else ("中" if bikes >= 3 else "低")
            plans.append(
                {
                    "source_grid_id": s.grid_id,
                    "target_grid_id": t.grid_id,
                    "source_lng": s.center_lng,
                    "source_lat": s.center_lat,
                    "target_lng": t.center_lng,
                    "target_lat": t.center_lat,
                    "dispatch_bikes": bikes,
                    "distance_km": round(distance, 2),
                    "priority": priority,
                    "reason": "目标区域预测借车需求较高，当前停放供给不足，建议从附近车辆过剩区域调入。",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            available -= bikes
            shortage_state[t.grid_id] = max(shortage_state[t.grid_id] - bikes, 0)
    return pd.DataFrame(plans)


def generate_and_save_dispatch(engine=None, threshold: float = 1.0) -> pd.DataFrame:
    predictions = query_df("SELECT * FROM prediction_results", engine=engine)
    plans = generate_dispatch_plan(predictions, threshold=threshold)
    replace_dispatch_plans(plans, engine=engine)
    return plans


def dispatch_gap_summary(engine=None, plans: pd.DataFrame | None = None) -> dict:
    """Return dispatch KPI values for the latest prediction and plan tables."""
    predictions = query_df("SELECT * FROM prediction_results", engine=engine)
    if plans is None:
        plans = query_df("SELECT * FROM dispatch_plans", engine=engine)

    if predictions.empty:
        return {
            "before_gap": 0.0,
            "after_gap": 0.0,
            "relief_rate": 0.0,
            "task_count": 0,
            "dispatch_bikes": 0,
            "total_distance": 0.0,
        }

    pred = predictions.copy()
    pred["predicted_start_count"] = pd.to_numeric(pred["predicted_start_count"], errors="coerce").fillna(0)
    pred["predicted_end_count"] = pd.to_numeric(pred["predicted_end_count"], errors="coerce").fillna(0)
    shortage = (pred["predicted_start_count"] - pred["predicted_end_count"]).clip(lower=0)
    before_gap = float(shortage.sum())

    dispatch_bikes = 0
    total_distance = 0.0
    task_count = 0
    if plans is not None and not plans.empty:
        dispatch_bikes = int(pd.to_numeric(plans["dispatch_bikes"], errors="coerce").fillna(0).sum())
        total_distance = float(pd.to_numeric(plans["distance_km"], errors="coerce").fillna(0).sum())
        task_count = int(len(plans))

    after_gap = max(before_gap - dispatch_bikes, 0.0)
    relief_rate = (before_gap - after_gap) / before_gap if before_gap > 0 else 0.0
    return {
        "before_gap": round(before_gap, 2),
        "after_gap": round(after_gap, 2),
        "relief_rate": round(relief_rate, 4),
        "task_count": task_count,
        "dispatch_bikes": dispatch_bikes,
        "total_distance": round(total_distance, 2),
    }
