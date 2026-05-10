import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN


RENAME_MAP = {
    "orderGuid": "order_guid",
    "bikeId": "bike_id",
    "rideDistance": "ride_distance",
    "rideTime": "ride_time",
    "startPointLng": "start_lng",
    "startPointLat": "start_lat",
    "endPointLng": "end_lng",
    "endPointLat": "end_lat",
    "startTime": "start_time",
    "endTime": "end_time",
    "rmqType": "rmq_type",
}


def _hash_user(value: object) -> str | None:
    if pd.isna(value) or value == "":
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def desensitize_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "phone" in df.columns:
        df = df.drop(columns=["phone"])
    if "userNewId" in df.columns:
        df["user_hash"] = df["userNewId"].map(_hash_user)
        df = df.drop(columns=["userNewId"])
    elif "user_hash" not in df.columns:
        df["user_hash"] = None
    return df


def normalize_order_frame(orders: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(orders)
    if df.empty:
        return df
    df = desensitize_orders(df)
    df = df.rename(columns=RENAME_MAP)
    if "order_guid" in df.columns:
        df = df.drop_duplicates(subset=["order_guid"])
    return df


def prepare_raw_orders(orders: list[dict], source: str = "sd_open_api") -> pd.DataFrame:
    clean_records = []
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for order in orders:
        record = dict(order)
        record.pop("phone", None)
        user_hash = _hash_user(record.pop("userNewId", None))
        normalized = normalize_order_frame([record])
        if normalized.empty:
            continue
        row = normalized.iloc[0].to_dict()
        row["user_hash"] = user_hash
        row["source"] = source
        row["crawl_time"] = crawl_time
        row["raw_json"] = json.dumps(record, ensure_ascii=False)
        clean_records.append(row)
    return pd.DataFrame(clean_records)


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["start_lng", "start_lat", "end_lng", "end_lat"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    mask = (
        df["start_lng"].between(LNG_MIN, LNG_MAX)
        & df["end_lng"].between(LNG_MIN, LNG_MAX)
        & df["start_lat"].between(LAT_MIN, LAT_MAX)
        & df["end_lat"].between(LAT_MIN, LAT_MAX)
    )
    return df[mask].copy()


def remove_abnormal_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ride_distance"] = pd.to_numeric(df.get("ride_distance"), errors="coerce")
    df["ride_time"] = pd.to_numeric(df.get("ride_time"), errors="coerce")
    mask = (df["ride_distance"] > 0) & (df["ride_time"] > 0)
    return df[mask].copy()


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df.get("start_time"), errors="coerce")
    df["end_time"] = pd.to_datetime(df.get("end_time"), errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df["start_date"] = df["start_time"].dt.strftime("%Y-%m-%d")
    df["end_date"] = df["end_time"].dt.strftime("%Y-%m-%d")
    df["start_hour"] = df["start_time"].dt.hour.astype(int)
    df["end_hour"] = df["end_time"].dt.hour.astype(int)
    df["weekday"] = df["start_time"].dt.weekday.astype(int)
    df["is_weekend"] = np.where(df["weekday"] >= 5, 1, 0)
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["end_time"] = df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = df.rename(columns=RENAME_MAP)
    df = desensitize_orders(df)
    if "order_guid" in df.columns:
        df = df.drop_duplicates(subset=["order_guid"])
    required = ["order_guid", "start_lng", "start_lat", "end_lng", "end_lat", "start_time", "end_time"]
    df = df.dropna(subset=[col for col in required if col in df.columns])
    df = validate_coordinates(df)
    df = remove_abnormal_orders(df)
    df = extract_time_features(df)
    return df
