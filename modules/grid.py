import math

import pandas as pd

from config.settings import GRID_SIZE


def get_grid_id(lng: float, lat: float, grid_size: float = GRID_SIZE) -> str:
    lng_idx = math.floor(float(lng) / grid_size)
    lat_idx = math.floor(float(lat) / grid_size)
    return f"{lng_idx}_{lat_idx}"


def get_grid_center(grid_id: str, grid_size: float = GRID_SIZE) -> tuple[float, float]:
    lng_idx, lat_idx = [int(x) for x in grid_id.split("_")]
    return (lng_idx + 0.5) * grid_size, (lat_idx + 0.5) * grid_size


def add_grid_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["start_grid_id"] = df.apply(lambda r: get_grid_id(r["start_lng"], r["start_lat"]), axis=1)
    df["end_grid_id"] = df.apply(lambda r: get_grid_id(r["end_lng"], r["end_lat"]), axis=1)
    return df


def aggregate_grid_hour(clean_df: pd.DataFrame) -> pd.DataFrame:
    if clean_df.empty:
        return pd.DataFrame()
    df = clean_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_time"])
    df["end_dt"] = pd.to_datetime(df["end_time"])
    df["start_stat_time"] = df["start_dt"].dt.floor("h")
    df["end_stat_time"] = df["end_dt"].dt.floor("h")

    start_agg = (
        df.groupby(["start_grid_id", "start_stat_time"])
        .size()
        .reset_index(name="start_count")
        .rename(columns={"start_grid_id": "grid_id", "start_stat_time": "stat_time"})
    )
    end_agg = (
        df.groupby(["end_grid_id", "end_stat_time"])
        .size()
        .reset_index(name="end_count")
        .rename(columns={"end_grid_id": "grid_id", "end_stat_time": "stat_time"})
    )
    stats = pd.merge(start_agg, end_agg, on=["grid_id", "stat_time"], how="outer").fillna(0)
    stats["start_count"] = stats["start_count"].astype(int)
    stats["end_count"] = stats["end_count"].astype(int)
    stats["date"] = pd.to_datetime(stats["stat_time"]).dt.strftime("%Y-%m-%d")
    stats["hour"] = pd.to_datetime(stats["stat_time"]).dt.hour.astype(int)
    stats["weekday"] = pd.to_datetime(stats["stat_time"]).dt.weekday.astype(int)
    stats["is_weekend"] = (stats["weekday"] >= 5).astype(int)
    centers = stats["grid_id"].map(get_grid_center)
    stats["center_lng"] = centers.map(lambda x: x[0])
    stats["center_lat"] = centers.map(lambda x: x[1])
    stats["stat_time"] = pd.to_datetime(stats["stat_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return stats
