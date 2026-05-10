from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from config.settings import DATABASE_URL, SQLITE_DB_PATH, ensure_directories


def get_sqlite_db_path() -> Path | None:
    """Return the configured SQLite file path, or None for non-file databases."""
    url = make_url(DATABASE_URL)
    if url.drivername != "sqlite":
        return None
    if not url.database or url.database == ":memory:":
        return None
    return Path(url.database)


def ensure_database_file() -> Path | None:
    """Create data/database/bike_hotspot.db before any table operation if needed."""
    ensure_directories()
    db_path = get_sqlite_db_path()
    if db_path is None:
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch(exist_ok=True)
    return db_path


def get_engine() -> Engine:
    ensure_database_file()
    return create_engine(DATABASE_URL, future=True)


def init_db(engine: Engine | None = None) -> None:
    ensure_database_file()
    engine = engine or get_engine()
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS raw_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_guid TEXT UNIQUE,
            bike_id TEXT,
            ride_distance REAL,
            ride_time REAL,
            start_lng REAL,
            start_lat REAL,
            end_lng REAL,
            end_lat REAL,
            start_time TEXT,
            end_time TEXT,
            rmq_type TEXT,
            user_hash TEXT,
            source TEXT,
            crawl_time TEXT,
            raw_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS clean_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_guid TEXT UNIQUE,
            bike_id TEXT,
            ride_distance REAL,
            ride_time REAL,
            start_lng REAL,
            start_lat REAL,
            end_lng REAL,
            end_lat REAL,
            start_time TEXT,
            end_time TEXT,
            start_date TEXT,
            end_date TEXT,
            start_hour INTEGER,
            end_hour INTEGER,
            weekday INTEGER,
            is_weekend INTEGER,
            start_grid_id TEXT,
            end_grid_id TEXT,
            crawl_time TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS grid_hour_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grid_id TEXT,
            stat_time TEXT,
            date TEXT,
            hour INTEGER,
            weekday INTEGER,
            is_weekend INTEGER,
            start_count INTEGER,
            end_count INTEGER,
            center_lng REAL,
            center_lat REAL,
            created_at TEXT,
            UNIQUE(grid_id, stat_time)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS hotspot_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grid_id TEXT,
            stat_time TEXT,
            end_count INTEGER,
            hotspot_level TEXT,
            hotspot_score REAL,
            center_lng REAL,
            center_lat REAL,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grid_id TEXT,
            predict_time TEXT,
            predicted_end_count REAL,
            predicted_start_count REAL,
            demand_gap REAL,
            risk_level TEXT,
            model_name TEXT,
            center_lng REAL,
            center_lat REAL,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dispatch_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_grid_id TEXT,
            target_grid_id TEXT,
            source_lng REAL,
            source_lat REAL,
            target_lng REAL,
            target_lat REAL,
            dispatch_bikes INTEGER,
            distance_km REAL,
            priority TEXT,
            reason TEXT,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS crawl_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_date TEXT,
            end_date TEXT,
            pages INTEGER,
            fetched_count INTEGER,
            inserted_count INTEGER,
            duplicate_count INTEGER,
            status TEXT,
            message TEXT,
            crawl_time TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            train_start_time TEXT,
            train_end_time TEXT,
            sample_count INTEGER,
            mae REAL,
            rmse REAL,
            r2 REAL,
            model_path TEXT,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS weather_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            temperature_mean REAL,
            precipitation_sum REAL,
            wind_speed_max REAL,
            weather_code INTEGER,
            source TEXT,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS holiday_calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            weekday INTEGER,
            is_weekend INTEGER,
            is_holiday INTEGER,
            holiday_name TEXT,
            created_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS system_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            status TEXT,
            created_at TEXT,
            last_login TEXT
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))
        existing_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(weather_daily)")).fetchall()]
        if "weather_code" not in existing_cols:
            conn.execute(text("ALTER TABLE weather_daily ADD COLUMN weather_code INTEGER"))
        prediction_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(prediction_results)")).fetchall()]
        if "demand_gap" not in prediction_cols:
            conn.execute(text("ALTER TABLE prediction_results ADD COLUMN demand_gap REAL"))


def initialize_database() -> Path | None:
    """Public bootstrap used by Streamlit pages and direct PyCharm runs."""
    engine = get_engine()
    init_db(engine)
    return get_sqlite_db_path()


RAW_COLUMNS = [
    "order_guid",
    "bike_id",
    "ride_distance",
    "ride_time",
    "start_lng",
    "start_lat",
    "end_lng",
    "end_lat",
    "start_time",
    "end_time",
    "rmq_type",
    "user_hash",
    "source",
    "crawl_time",
    "raw_json",
]

CLEAN_COLUMNS = [
    "order_guid",
    "bike_id",
    "ride_distance",
    "ride_time",
    "start_lng",
    "start_lat",
    "end_lng",
    "end_lat",
    "start_time",
    "end_time",
    "start_date",
    "end_date",
    "start_hour",
    "end_hour",
    "weekday",
    "is_weekend",
    "start_grid_id",
    "end_grid_id",
    "crawl_time",
]


def _insert_ignore_df(engine: Engine, df: pd.DataFrame, table: str, columns: list[str]) -> tuple[int, int]:
    if df.empty:
        return 0, 0
    rows = df.reindex(columns=columns).where(pd.notna(df), None).to_dict("records")
    placeholders = ", ".join([f":{c}" for c in columns])
    sql = text(f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})")
    inserted = 0
    with engine.begin() as conn:
        for row in rows:
            result = conn.execute(sql, row)
            inserted += result.rowcount or 0
    return inserted, len(rows) - inserted


def insert_raw_orders(df: pd.DataFrame, engine: Engine | None = None) -> tuple[int, int]:
    engine = engine or get_engine()
    return _insert_ignore_df(engine, df, "raw_orders", RAW_COLUMNS)


def insert_clean_orders(df: pd.DataFrame, engine: Engine | None = None) -> tuple[int, int]:
    engine = engine or get_engine()
    return _insert_ignore_df(engine, df, "clean_orders", CLEAN_COLUMNS)


def replace_grid_hour_stats(df: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    if df.empty:
        return 0
    write_df = df.copy()
    write_df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    columns = [
        "grid_id",
        "stat_time",
        "date",
        "hour",
        "weekday",
        "is_weekend",
        "start_count",
        "end_count",
        "center_lng",
        "center_lat",
        "created_at",
    ]
    sql = text(
        f"""
        INSERT OR REPLACE INTO grid_hour_stats ({', '.join(columns)})
        VALUES ({', '.join([f':{c}' for c in columns])})
        """
    )
    rows = write_df.reindex(columns=columns).where(pd.notna(write_df), None).to_dict("records")
    with engine.begin() as conn:
        for row in rows:
            conn.execute(sql, row)
    return len(rows)


def replace_hotspots(df: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM hotspot_results"))
    if df.empty:
        return 0
    df.to_sql("hotspot_results", engine, if_exists="append", index=False)
    return len(df)


def replace_predictions(df: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM prediction_results"))
    if df.empty:
        return 0
    columns = [
        "grid_id",
        "predict_time",
        "predicted_end_count",
        "predicted_start_count",
        "demand_gap",
        "risk_level",
        "model_name",
        "center_lng",
        "center_lat",
        "created_at",
    ]
    write_df = df.copy()
    if "demand_gap" not in write_df.columns:
        write_df["demand_gap"] = pd.to_numeric(write_df.get("predicted_start_count"), errors="coerce").fillna(0) - pd.to_numeric(write_df.get("predicted_end_count"), errors="coerce").fillna(0)
    write_df.reindex(columns=columns).to_sql("prediction_results", engine, if_exists="append", index=False)
    return len(write_df)


def replace_dispatch_plans(df: pd.DataFrame, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM dispatch_plans"))
    if df.empty:
        return 0
    df.to_sql("dispatch_plans", engine, if_exists="append", index=False)
    return len(df)


def log_crawl(start_date: str, end_date: str, pages: int, fetched_count: int, inserted_count: int, duplicate_count: int, status: str, message: str, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO crawl_logs
                (start_date, end_date, pages, fetched_count, inserted_count, duplicate_count, status, message, crawl_time)
                VALUES (:start_date, :end_date, :pages, :fetched_count, :inserted_count, :duplicate_count, :status, :message, :crawl_time)
                """
            ),
            {
                "start_date": start_date,
                "end_date": end_date,
                "pages": pages,
                "fetched_count": fetched_count,
                "inserted_count": inserted_count,
                "duplicate_count": duplicate_count,
                "status": status,
                "message": message,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


def log_model(row: dict, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    pd.DataFrame([row]).to_sql("model_logs", engine, if_exists="append", index=False)


def query_df(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    try:
        return pd.read_sql_query(text(sql), engine, params=params or {})
    except Exception:
        return pd.DataFrame()


def table_count(table: str, engine: Engine | None = None) -> int:
    df = query_df(f"SELECT COUNT(*) AS c FROM {table}", engine=engine)
    return int(df["c"].iloc[0]) if not df.empty else 0


def system_summary(engine: Engine | None = None) -> dict:
    engine = engine or get_engine()
    summary = {
        "raw_orders": table_count("raw_orders", engine),
        "clean_orders": table_count("clean_orders", engine),
        "bike_count": 0,
        "hotspot_high": 0,
        "prediction_high": 0,
        "dispatch_bikes": 0,
        "last_crawl": "",
        "last_model_time": "",
    }
    bike_df = query_df("SELECT COUNT(DISTINCT bike_id) AS c FROM clean_orders", engine=engine)
    if not bike_df.empty:
        summary["bike_count"] = int(bike_df["c"].iloc[0] or 0)
    hot_df = query_df("SELECT COUNT(*) AS c FROM hotspot_results WHERE hotspot_level='高'", engine=engine)
    if not hot_df.empty:
        summary["hotspot_high"] = int(hot_df["c"].iloc[0] or 0)
    pred_df = query_df("SELECT COUNT(*) AS c FROM prediction_results WHERE risk_level='高'", engine=engine)
    if not pred_df.empty:
        summary["prediction_high"] = int(pred_df["c"].iloc[0] or 0)
    dis_df = query_df("SELECT SUM(dispatch_bikes) AS c FROM dispatch_plans", engine=engine)
    if not dis_df.empty:
        summary["dispatch_bikes"] = int(dis_df["c"].iloc[0] or 0)
    crawl_df = query_df("SELECT crawl_time FROM crawl_logs ORDER BY id DESC LIMIT 1", engine=engine)
    if not crawl_df.empty:
        summary["last_crawl"] = str(crawl_df["crawl_time"].iloc[0])
    model_df = query_df("SELECT created_at FROM model_logs ORDER BY id DESC LIMIT 1", engine=engine)
    if not model_df.empty:
        summary["last_model_time"] = str(model_df["created_at"].iloc[0])
    return summary
