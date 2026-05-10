import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, require_login
from modules.database import get_engine, init_db, query_df, table_count
from modules.ui import show_page_error


st.set_page_config(page_title="数据管理", layout="wide")


TABLES = [
    ("raw_orders", "原始脱敏订单"),
    ("clean_orders", "清洗后订单"),
    ("grid_hour_stats", "网格小时统计"),
    ("prediction_results", "预测结果"),
    ("dispatch_plans", "调度方案"),
]


def export_table_button(table: str, label: str, engine) -> None:
    df = query_df(f"SELECT * FROM {table}", engine=engine)
    if df.empty:
        st.caption(f"{label}暂无可导出数据。")
        return
    st.download_button(
        f"导出{label} CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        f"{table}.csv",
        "text/csv",
        use_container_width=True,
    )


try:
    require_login([ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER])
    engine = get_engine()
    init_db(engine)

    st.title("数据管理")
    st.caption("系统不保存手机号原文；用户标识仅以哈希形式进入原始脱敏表，清洗表只保留时间、空间和骑行特征。")

    counts = {table: table_count(table, engine) for table, _ in TABLES}
    cols = st.columns(len(TABLES))
    for col, (table, label) in zip(cols, TABLES):
        col.metric(label, counts[table])

    range_df = query_df(
        "SELECT MIN(start_time) AS min_time, MAX(end_time) AS max_time, COUNT(*) AS orders FROM clean_orders",
        engine=engine,
    )
    if not range_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("有效订单数", int(range_df["orders"].iloc[0] or 0))
        c2.metric("最早订单时间", range_df["min_time"].iloc[0] or "暂无")
        c3.metric("最晚订单时间", range_df["max_time"].iloc[0] or "暂无")

    latest_log = query_df("SELECT * FROM crawl_logs ORDER BY id DESC LIMIT 1", engine=engine)
    st.subheader("最近一次采集日志")
    if latest_log.empty:
        st.info("暂无采集日志。请先在“数据采集”页面执行采集。")
    else:
        show_log = latest_log.rename(
            columns={
                "start_date": "开始时间",
                "end_date": "结束时间",
                "pages": "页数",
                "fetched_count": "接口返回订单数",
                "inserted_count": "新增订单数",
                "duplicate_count": "重复订单数",
                "status": "状态",
                "message": "说明",
                "crawl_time": "采集时间",
            }
        )
        st.dataframe(show_log, use_container_width=True, hide_index=True)

    tab1, tab2, tab3, tab4 = st.tabs(["清洗后订单", "原始脱敏订单", "数据库导出", "字段说明"])

    with tab1:
        clean = query_df("SELECT * FROM clean_orders ORDER BY id DESC LIMIT 500", engine=engine)
        if clean.empty:
            st.info("暂无清洗后订单。采集完成后系统会自动清洗并写入该表。")
        else:
            st.dataframe(clean, use_container_width=True, hide_index=True)
            st.download_button(
                "导出当前清洗订单 CSV",
                clean.to_csv(index=False).encode("utf-8-sig"),
                "clean_orders_preview.csv",
                "text/csv",
            )

    with tab2:
        raw = query_df(
            """
            SELECT id, order_guid, bike_id, ride_distance, ride_time,
                   start_lng, start_lat, end_lng, end_lat,
                   start_time, end_time, rmq_type, user_hash, source, crawl_time
            FROM raw_orders
            ORDER BY id DESC
            LIMIT 500
            """,
            engine=engine,
        )
        if raw.empty:
            st.info("暂无原始脱敏订单。")
        else:
            st.dataframe(raw, use_container_width=True, hide_index=True)
            st.download_button(
                "导出当前原始脱敏订单 CSV",
                raw.to_csv(index=False).encode("utf-8-sig"),
                "raw_orders_desensitized_preview.csv",
                "text/csv",
            )

    with tab3:
        st.markdown("以下导出均来自数据库表，不包含手机号原文。")
        export_cols = st.columns(5)
        for col, (table, label) in zip(export_cols, TABLES):
            with col:
                export_table_button(table, label, engine)

    with tab4:
        st.markdown(
            """
- `order_guid`：订单唯一编号，用于增量去重，数据库设置唯一约束。
- `bike_id`：车辆编号，用于车辆使用频率和维修预警分析。
- `ride_distance` / `ride_time`：接口返回的骑行距离与骑行时长。
- `start_lng/start_lat`、`end_lng/end_lat`：起点和终点经纬度。
- `start_grid_id/end_grid_id`：网格化后的起点和终点区域编号。
- `phone`：采集后立即删除，不入库、不展示、不导出。
- `userNewId`：不保存原文；如需保留，只保存 SHA256 哈希到 `user_hash`。
"""
        )
except Exception as exc:
    show_page_error(exc, "数据管理页面加载失败，请检查数据库文件和表结构是否正常。")
