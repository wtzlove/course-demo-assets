from datetime import date, datetime, timedelta

import pandas as pd
import requests
from sqlalchemy import text

from config.settings import LIANGSHAN_CENTER
from modules.database import get_engine, query_df


HOLIDAYS_2026 = {
    "2026-01-01": "元旦",
    "2026-02-16": "春节",
    "2026-02-17": "春节",
    "2026-02-18": "春节",
    "2026-02-19": "春节",
    "2026-02-20": "春节",
    "2026-02-21": "春节",
    "2026-02-22": "春节",
    "2026-04-04": "清明节",
    "2026-04-05": "清明节",
    "2026-04-06": "清明节",
    "2026-05-01": "劳动节",
    "2026-05-02": "劳动节",
    "2026-05-03": "劳动节",
    "2026-05-04": "劳动节",
    "2026-05-05": "劳动节",
    "2026-06-19": "端午节",
    "2026-06-20": "端午节",
    "2026-06-21": "端午节",
    "2026-09-25": "中秋节",
    "2026-09-26": "中秋节",
    "2026-09-27": "中秋节",
    "2026-10-01": "国庆节",
    "2026-10-02": "国庆节",
    "2026-10-03": "国庆节",
    "2026-10-04": "国庆节",
    "2026-10-05": "国庆节",
    "2026-10-06": "国庆节",
    "2026-10-07": "国庆节",
}


def upsert_holiday_calendar(start_date: str, end_date: str, engine=None) -> int:
    engine = engine or get_engine()
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    rows = []
    current = start
    while current <= end:
        text_date = current.strftime("%Y-%m-%d")
        weekday = current.weekday()
        rows.append(
            {
                "date": text_date,
                "weekday": weekday,
                "is_weekend": 1 if weekday >= 5 else 0,
                "is_holiday": 1 if text_date in HOLIDAYS_2026 else 0,
                "holiday_name": HOLIDAYS_2026.get(text_date, ""),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        current += timedelta(days=1)

    sql = text(
        """
        INSERT OR REPLACE INTO holiday_calendar
        (date, weekday, is_weekend, is_holiday, holiday_name, created_at)
        VALUES (:date, :weekday, :is_weekend, :is_holiday, :holiday_name, :created_at)
        """
    )
    with engine.begin() as conn:
        for row in rows:
            conn.execute(sql, row)
    return len(rows)


def fetch_open_meteo_daily(start_date: str, end_date: str) -> pd.DataFrame:
    lat = LIANGSHAN_CENTER[1]
    lng = LIANGSHAN_CENTER[0]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "Asia/Shanghai",
    }
    response = requests.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    daily = response.json().get("daily", {})
    if not daily:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temperature_mean": daily.get("temperature_2m_mean", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "wind_speed_max": daily.get("wind_speed_10m_max", []),
            "weather_code": daily.get("weather_code", [0] * len(daily.get("time", []))),
        }
    )


def fetch_open_meteo_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    lat = LIANGSHAN_CENTER[1]
    lng = LIANGSHAN_CENTER[0]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "Asia/Shanghai",
    }
    response = requests.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    daily = response.json().get("daily", {})
    if not daily:
        return pd.DataFrame()
    max_temp = pd.Series(daily.get("temperature_2m_max", []), dtype="float64")
    min_temp = pd.Series(daily.get("temperature_2m_min", []), dtype="float64")
    return pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temperature_mean": ((max_temp + min_temp) / 2).round(1).tolist(),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "wind_speed_max": daily.get("wind_speed_10m_max", []),
            "weather_code": daily.get("weather_code", []),
        }
    )


def fetch_weather_by_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    today = datetime.now().date()
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    frames = []
    if start <= today:
        archive_end = min(end, today)
        frames.append(fetch_open_meteo_daily(start.strftime("%Y-%m-%d"), archive_end.strftime("%Y-%m-%d")))
    if end > today:
        forecast_start = max(start, today + timedelta(days=1))
        frames.append(fetch_open_meteo_forecast(forecast_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")


def upsert_weather_daily(df: pd.DataFrame, engine=None, source: str = "Open-Meteo") -> int:
    engine = engine or get_engine()
    if df.empty:
        return 0
    rows = df.copy()
    if "weather_code" not in rows.columns:
        rows["weather_code"] = 0
    rows["source"] = source
    rows["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = text(
        """
        INSERT OR REPLACE INTO weather_daily
        (date, temperature_mean, precipitation_sum, wind_speed_max, weather_code, source, created_at)
        VALUES (:date, :temperature_mean, :precipitation_sum, :wind_speed_max, :weather_code, :source, :created_at)
        """
    )
    with engine.begin() as conn:
        for row in rows.to_dict("records"):
            conn.execute(sql, row)
    return len(rows)


def collect_environment_data(start_date: str, end_date: str, engine=None) -> dict:
    engine = engine or get_engine()
    holiday_rows = upsert_holiday_calendar(start_date, end_date, engine=engine)
    weather_rows = 0
    weather_error = ""
    try:
        weather_df = fetch_weather_by_date_range(start_date, end_date)
        weather_rows = upsert_weather_daily(weather_df, engine=engine)
    except Exception as exc:
        weather_error = str(exc)
    return {"holiday_rows": holiday_rows, "weather_rows": weather_rows, "weather_error": weather_error}


def load_environment_features(engine=None) -> pd.DataFrame:
    weather = query_df("SELECT * FROM weather_daily", engine=engine)
    holiday = query_df("SELECT * FROM holiday_calendar", engine=engine)
    if weather.empty and holiday.empty:
        return pd.DataFrame()
    if weather.empty:
        return holiday
    if holiday.empty:
        return weather
    return pd.merge(weather, holiday, on="date", how="outer")


def ensure_environment_for_date(target_date: str, engine=None) -> dict:
    engine = engine or get_engine()
    target = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    holiday_rows = upsert_holiday_calendar(target, target, engine=engine)
    weather_exists = query_df("SELECT COUNT(*) AS c FROM weather_daily WHERE date=:date", {"date": target}, engine=engine)
    weather_rows = 0
    weather_error = ""
    if weather_exists.empty or int(weather_exists["c"].iloc[0]) == 0:
        try:
            weather_df = fetch_weather_by_date_range(target, target)
            weather_rows = upsert_weather_daily(weather_df, engine=engine)
        except Exception as exc:
            weather_error = str(exc)
    return {"holiday_rows": holiday_rows, "weather_rows": weather_rows, "weather_error": weather_error}


def environment_for_date(target_date: str, engine=None) -> dict:
    engine = engine or get_engine()
    target = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    env = load_environment_features(engine=engine)
    if env.empty:
        return {
            "temperature_mean": 0,
            "precipitation_sum": 0,
            "wind_speed_max": 0,
            "is_holiday": 0,
            "is_weekend": 1 if pd.to_datetime(target).weekday() >= 5 else 0,
        }
    row = env[env["date"] == target]
    if row.empty:
        return {
            "temperature_mean": 0,
            "precipitation_sum": 0,
            "wind_speed_max": 0,
            "is_holiday": 0,
            "is_weekend": 1 if pd.to_datetime(target).weekday() >= 5 else 0,
        }
    record = row.iloc[0].to_dict()
    return {
        "temperature_mean": float(record.get("temperature_mean") or 0),
        "precipitation_sum": float(record.get("precipitation_sum") or 0),
        "wind_speed_max": float(record.get("wind_speed_max") or 0),
        "is_holiday": int(record.get("is_holiday") or 0),
        "is_weekend": int(record.get("is_weekend") or 0),
    }
