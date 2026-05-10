import pandas as pd

from modules.api_client import fetch_range
from modules.cleaner import clean_orders, prepare_raw_orders
from modules.database import (
    get_engine,
    init_db,
    insert_clean_orders,
    insert_raw_orders,
    log_crawl,
    query_df,
    replace_grid_hour_stats,
    replace_hotspots,
)
from modules.grid import add_grid_columns, aggregate_grid_hour
from modules.hotspot import identify_hotspots


def rebuild_derived_tables(engine=None) -> dict:
    engine = engine or get_engine()
    clean_df = query_df("SELECT * FROM clean_orders", engine=engine)
    if clean_df.empty:
        replace_hotspots(pd.DataFrame(), engine=engine)
        return {"grid_stats": 0, "hotspots": 0}
    stats = aggregate_grid_hour(clean_df)
    grid_rows = replace_grid_hour_stats(stats, engine=engine)
    hotspots = identify_hotspots(stats)
    hotspot_rows = replace_hotspots(hotspots, engine=engine)
    return {"grid_stats": grid_rows, "hotspots": hotspot_rows}


def ingest_orders(orders: list[dict], start_date: str, end_date: str, pages: int, engine=None) -> dict:
    engine = engine or get_engine()
    init_db(engine)
    raw_df = prepare_raw_orders(orders)
    inserted_raw, duplicate_raw = insert_raw_orders(raw_df, engine=engine)

    clean_df = clean_orders(raw_df)
    clean_df = add_grid_columns(clean_df)
    inserted_clean, _ = insert_clean_orders(clean_df, engine=engine)
    derived = rebuild_derived_tables(engine=engine)
    log_crawl(
        start_date,
        end_date,
        pages,
        len(orders),
        inserted_raw,
        duplicate_raw,
        "success",
        f"清洗入库 {inserted_clean} 条，更新网格 {derived['grid_stats']} 条，热点 {derived['hotspots']} 条。",
        engine=engine,
    )
    return {
        "fetched_count": len(orders),
        "inserted_count": inserted_raw,
        "duplicate_count": duplicate_raw,
        "clean_inserted_count": inserted_clean,
        "pages": pages,
        **derived,
    }


def crawl_and_update(start_date: str, end_date: str, max_pages: int = 5, engine=None) -> dict:
    engine = engine or get_engine()
    init_db(engine)
    try:
        payload = fetch_range(start_date, end_date, max_pages=max_pages)
        return ingest_orders(payload["orders"], start_date, end_date, payload["pages"], engine=engine)
    except Exception as exc:
        log_crawl(start_date, end_date, 0, 0, 0, 0, "failed", str(exc), engine=engine)
        raise
