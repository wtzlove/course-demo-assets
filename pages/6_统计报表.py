import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, require_login
from modules.database import get_engine, init_db, query_df
from modules.report import (
    generate_system_summary_text,
    get_bike_frequency,
    get_daily_orders,
    get_hourly_orders,
    get_ride_metrics,
    get_top_end_grids,
    get_top_start_grids,
    make_daily_chart,
    make_distribution_chart,
    make_hourly_chart,
    make_maintenance_warning_chart,
)
from modules.ui import show_page_error


st.set_page_config(page_title="统计报表", layout="wide")


try:
    require_login([ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER])
    engine = get_engine()
    init_db(engine)

    st.title("统计报表")

    st.subheader("系统运行摘要")
    st.write(generate_system_summary_text(engine))

    daily = get_daily_orders(engine)
    hourly = get_hourly_orders(engine)
    ride_metrics = get_ride_metrics(engine)

    c1, c2 = st.columns(2)
    with c1:
        chart = make_daily_chart(daily)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("暂无每日订单量数据。")
    with c2:
        chart = make_hourly_chart(hourly)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("暂无小时订单量数据。")

    c3, c4 = st.columns(2)
    with c3:
        chart = make_distribution_chart(
            ride_metrics,
            "骑行直线距离km",
            "骑行直线距离分布",
            bins=[0, 0.2, 0.5, 1, 1.5, 2, 3, 5, 8, float("inf")],
            suffix="km",
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("暂无骑行距离分布数据。")
    with c4:
        chart = make_distribution_chart(
            ride_metrics,
            "骑行时长分钟",
            "骑行时长分布",
            bins=[0, 3, 5, 10, 15, 20, 30, 45, 60, float("inf")],
            suffix="分钟",
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("暂无骑行时长分布数据。")

    st.subheader("热点区域排名")
    h1, h2 = st.columns(2)
    with h1:
        start_top = get_top_start_grids(engine).rename(columns={"orders": "订单数量", "grid_id": "网格编号"})
        if start_top.empty:
            st.info("暂无起点热点数据。")
        else:
            st.markdown("**起点热点 Top 10**")
            st.dataframe(start_top, use_container_width=True, hide_index=True, height=390)
    with h2:
        end_top = get_top_end_grids(engine).rename(columns={"orders": "订单数量", "grid_id": "网格编号"})
        if end_top.empty:
            st.info("暂无终点热点数据。")
        else:
            st.markdown("**终点热点 Top 10**")
            st.dataframe(end_top, use_container_width=True, hide_index=True, height=390)

    st.subheader("车辆使用频率 Top 10")
    bike_freq = get_bike_frequency(engine)
    left, right = st.columns([1.35, 0.65])
    with left:
        chart = make_maintenance_warning_chart(bike_freq, threshold=50)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("暂无车辆使用频率数据。")
        st.caption("维修规则：右侧 Top10 车辆中，累计订单数达到 50 次建议检查维护；完成检查后可在运营记录中归零重新统计。")
    with right:
        if bike_freq.empty:
            st.info("暂无车辆使用频率数据。")
        else:
            st.dataframe(
                bike_freq.rename(columns={"bike_id": "车辆编号", "orders": "订单数量"}),
                use_container_width=True,
                hide_index=True,
                height=430,
            )

    st.subheader("采集日志")
    crawl_logs = query_df("SELECT * FROM crawl_logs ORDER BY id DESC LIMIT 50", engine=engine)
    if crawl_logs.empty:
        st.info("暂无采集日志。")
    else:
        st.dataframe(crawl_logs, use_container_width=True, hide_index=True)

    st.subheader("模型训练日志")
    model_logs = query_df("SELECT * FROM model_logs ORDER BY id DESC LIMIT 50", engine=engine)
    if model_logs.empty:
        st.info("暂无模型训练日志。")
    else:
        st.dataframe(model_logs, use_container_width=True, hide_index=True)
except Exception as exc:
    show_page_error(exc, "统计报表页面加载失败，请检查订单数据或报表统计函数。")
