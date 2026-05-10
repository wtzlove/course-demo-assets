from datetime import datetime

import numpy as np
import pandas as pd


def calculate_hotspot_score(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    end_max = max(df["end_count"].max(), 1)
    start_max = max(df["start_count"].max(), 1)
    return (0.7 * df["end_count"] / end_max + 0.3 * df["start_count"] / start_max) * 100


def identify_hotspots(grid_hour_stats: pd.DataFrame) -> pd.DataFrame:
    if grid_hour_stats.empty:
        return pd.DataFrame()
    df = grid_hour_stats.copy()
    q80 = df["end_count"].quantile(0.8)
    q50 = df["end_count"].quantile(0.5)
    df["hotspot_level"] = np.select(
        [df["end_count"] >= q80, df["end_count"] >= q50],
        ["高", "中"],
        default="低",
    )
    df["hotspot_score"] = calculate_hotspot_score(df).round(2)
    df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df[
        [
            "grid_id",
            "stat_time",
            "end_count",
            "hotspot_level",
            "hotspot_score",
            "center_lng",
            "center_lat",
            "created_at",
        ]
    ]


def get_top_hotspots(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(["hotspot_score", "end_count"], ascending=False).head(top_n)
