from __future__ import annotations

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


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    for col in ["predicted_start_count", "predicted_end_count", "center_lng", "center_lat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["grid_id", "predicted_start_count", "predicted_end_count", "center_lng", "center_lat"])
    df["demand_gap"] = pd.to_numeric(
        df.get("demand_gap", df["predicted_start_count"] - df["predicted_end_count"]),
        errors="coerce",
    ).fillna(df["predicted_start_count"] - df["predicted_end_count"])
    return df


def _priority(target_gap: float, distance_km: float, bikes: int) -> str:
    if target_gap >= 5 or bikes >= 5 or distance_km <= 1.0:
        return "高"
    if target_gap >= 3 or bikes >= 3 or distance_km <= 2.5:
        return "中"
    return "低"


def generate_dispatch_plan(predictions: pd.DataFrame, threshold: float = 1.0, max_tasks: int = 30) -> pd.DataFrame:
    """Generate a heuristic plan from surplus grid areas to shortage grid areas."""
    if predictions.empty:
        return pd.DataFrame()

    df = _prepare_predictions(predictions)
    if df.empty:
        return pd.DataFrame()

    shortage = df[df["demand_gap"] > threshold].sort_values("demand_gap", ascending=False).copy()
    surplus = df[df["demand_gap"] < -threshold].sort_values("demand_gap", ascending=True).copy()
    if shortage.empty or surplus.empty:
        return pd.DataFrame()

    shortage_state = {str(row.grid_id): float(row.demand_gap) for row in shortage.itertuples()}
    surplus_state = {str(row.grid_id): abs(float(row.demand_gap)) for row in surplus.itertuples()}
    plans: list[dict] = []

    # Priority is driven by target shortage first; distance is used to choose nearby supply.
    for target in shortage.itertuples():
        target_id = str(target.grid_id)
        remaining_need = shortage_state.get(target_id, 0.0)
        if remaining_need <= 0:
            continue

        candidates = []
        for source in surplus.itertuples():
            source_id = str(source.grid_id)
            available = surplus_state.get(source_id, 0.0)
            if available <= 0 or source_id == target_id:
                continue
            distance = haversine_km(source.center_lng, source.center_lat, target.center_lng, target.center_lat)
            # Large target gaps should be handled first, but nearby surplus reduces operational cost.
            score = distance / max(remaining_need, 1.0)
            candidates.append((score, distance, source, available))

        for _, distance, source, available in sorted(candidates, key=lambda item: item[0]):
            if remaining_need <= 0 or len(plans) >= max_tasks:
                break
            bikes = int(min(round(available), round(remaining_need)))
            bikes = max(0, bikes)
            if bikes <= 0:
                continue

            priority = _priority(float(target.demand_gap), float(distance), bikes)
            plans.append(
                {
                    "source_grid_id": source.grid_id,
                    "target_grid_id": target.grid_id,
                    "source_lng": float(source.center_lng),
                    "source_lat": float(source.center_lat),
                    "target_lng": float(target.center_lng),
                    "target_lat": float(target.center_lat),
                    "dispatch_bikes": bikes,
                    "distance_km": round(float(distance), 2),
                    "priority": priority,
                    "reason": (
                        "目标网格区域预测借车需求高于还车供给，形成短时缺车风险；"
                        "调出区域预测还车供给相对充足，且空间距离较近，建议进行车辆调入。"
                    ),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            surplus_state[str(source.grid_id)] = max(available - bikes, 0.0)
            remaining_need = max(remaining_need - bikes, 0.0)
            shortage_state[target_id] = remaining_need

    return pd.DataFrame(plans)


def generate_and_save_dispatch(engine=None, threshold: float = 1.0) -> pd.DataFrame:
    predictions = query_df("SELECT * FROM prediction_results", engine=engine)
    plans = generate_dispatch_plan(predictions, threshold=threshold)
    replace_dispatch_plans(plans, engine=engine)
    return plans


def dispatch_gap_summary(engine=None, plans: pd.DataFrame | None = None) -> dict:
    """Summarize dispatch effect for graduation-design evaluation."""
    predictions = query_df("SELECT * FROM prediction_results", engine=engine)
    if plans is None:
        plans = query_df("SELECT * FROM dispatch_plans", engine=engine)

    if predictions.empty:
        return {
            "before_gap": 0.0,
            "dispatch_bikes": 0,
            "after_gap": 0.0,
            "relief_rate": 0.0,
            "task_count": 0,
            "total_distance": 0.0,
            "avg_distance": 0.0,
            "high_priority_tasks": 0,
        }

    pred = _prepare_predictions(predictions)
    shortage = pred["demand_gap"].clip(lower=0)
    before_gap = float(shortage.sum())

    dispatch_bikes = 0
    total_distance = 0.0
    task_count = 0
    high_priority_tasks = 0
    if plans is not None and not plans.empty:
        dispatch_bikes = int(pd.to_numeric(plans["dispatch_bikes"], errors="coerce").fillna(0).clip(lower=0).sum())
        total_distance = float(pd.to_numeric(plans["distance_km"], errors="coerce").fillna(0).clip(lower=0).sum())
        task_count = int(len(plans))
        high_priority_tasks = int((plans["priority"].astype(str) == "高").sum())

    after_gap = max(before_gap - dispatch_bikes, 0.0)
    relief_rate = (before_gap - after_gap) / before_gap if before_gap > 0 else 0.0
    avg_distance = total_distance / task_count if task_count else 0.0
    return {
        "before_gap": round(before_gap, 2),
        "dispatch_bikes": dispatch_bikes,
        "after_gap": round(after_gap, 2),
        "relief_rate": round(relief_rate, 4),
        "task_count": task_count,
        "total_distance": round(total_distance, 2),
        "avg_distance": round(avg_distance, 2),
        "high_priority_tasks": high_priority_tasks,
    }


def dispatch_explanation(summary: dict) -> str:
    if summary.get("task_count", 0) <= 0:
        return "当前预测结果中缺车区与余车区匹配不足，暂未生成调度任务。"
    return (
        f"本次调度优先缓解高缺口网格区域，建议从附近余车区域调入车辆，"
        f"共生成 {summary['task_count']} 条任务，建议调度 {summary['dispatch_bikes']} 辆，"
        f"预计可缓解 {summary['relief_rate'] * 100:.1f}% 的短时供需缺口。"
    )
