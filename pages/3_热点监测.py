from datetime import datetime

import plotly.express as px
import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, require_login
from modules.database import get_engine, init_db, query_df
from modules.geo_context import add_area_names
from modules.map_view import hotspot_map, order_heatmap
from modules.streamlit_map import render_map
from modules.trend import build_hourly_timeseries, forecast_next_hours, make_hourly_trend_chart, trend_summary
from modules.workflow import rebuild_derived_tables


st.set_page_config(page_title="热点监测", layout="wide")
require_login([ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER])
engine = get_engine()
init_db(engine)

st.title("热点监测")

if st.button("重新计算热点", use_container_width=True):
    with st.spinner("正在重新计算网格统计和热点结果..."):
        result = rebuild_derived_tables(engine)
    st.session_state["hotspot_rebuild_message"] = f"已更新网格统计 {result['grid_stats']} 条，热点结果 {result['hotspots']} 条。"
    st.rerun()

if st.session_state.get("hotspot_rebuild_message"):
    st.success(st.session_state.pop("hotspot_rebuild_message"))

date_range = query_df("SELECT MIN(start_date) AS min_date, MAX(start_date) AS max_date FROM clean_orders", engine=engine)
if date_range.empty or not date_range["min_date"].iloc[0]:
    st.info("暂无订单数据。请先在“数据采集”页面采集数据。")
    st.stop()

fixed_start = datetime.strptime(str(date_range["min_date"].iloc[0]), "%Y-%m-%d").date()
default_end = datetime.strptime(str(date_range["max_date"].iloc[0]), "%Y-%m-%d").date()

c1, c2 = st.columns(2)
c1.date_input("开始日期（数据库最早采集日期，固定）", value=fixed_start, disabled=True)
end_date = c2.date_input("结束日期", value=default_end, min_value=fixed_start)
if end_date < fixed_start:
    st.warning("结束日期不能早于开始日期。")
    st.stop()

params = {"start": fixed_start.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")}
orders = query_df(
    "SELECT * FROM clean_orders WHERE start_date BETWEEN :start AND :end ORDER BY start_time DESC",
    params=params,
    engine=engine,
)

st.caption(
    "热点排行榜表示在所选时间范围内，哪些网格的结束订单数量和停放压力更高。"
    "结束订单越多，说明车辆更容易在该区域集中停放；开始订单越多，说明该区域用车需求更强。"
)

view = st.radio("监测视图", ["终点热力图", "起点热力图", "热点排行榜", "小时趋势"], horizontal=True)

if view == "终点热力图":
    st.caption(f"终点有效坐标点：{orders[['end_lat', 'end_lng']].dropna().shape[0]} 个")
    render_map(order_heatmap(orders, "end"), height=560, key="end_heatmap")

elif view == "起点热力图":
    st.caption(f"起点有效坐标点：{orders[['start_lat', 'start_lng']].dropna().shape[0]} 个")
    render_map(order_heatmap(orders, "start"), height=560, key="start_heatmap")

elif view == "热点排行榜":
    hotspots = query_df(
        """
        SELECT h.*, COALESCE(g.start_count, 0) AS start_count
        FROM hotspot_results h
        LEFT JOIN grid_hour_stats g ON h.grid_id = g.grid_id AND h.stat_time = g.stat_time
        WHERE substr(h.stat_time, 1, 10) BETWEEN :start AND :end
        ORDER BY h.hotspot_score DESC
        LIMIT 120
        """,
        params=params,
        engine=engine,
    )
    if hotspots.empty:
        st.info("暂无热点结果。")
    else:
        render_map(hotspot_map(hotspots.head(80)), height=460, key="hotspot_bike_map")
        table = add_area_names(hotspots.head(20))
        table = table.rename(
            columns={
                "grid_id": "网格编号",
                "stat_time": "统计时间",
                "start_count": "开始订单数量",
                "end_count": "结束订单数量",
                "hotspot_level": "热点等级",
                "hotspot_score": "热点得分",
                "center_lng": "中心经度",
                "center_lat": "中心纬度",
            }
        )
        display_columns = [
            "区域名称",
            "区域类型",
            "距地标km",
            "网格编号",
            "统计时间",
            "开始订单数量",
            "结束订单数量",
            "热点等级",
            "热点得分",
        ]
        st.dataframe(table[display_columns], use_container_width=True, hide_index=True)

elif view == "小时趋势":
    stats = query_df(
        """
        SELECT stat_time, hour, start_count, end_count
        FROM grid_hour_stats
        WHERE date BETWEEN :start AND :end
        ORDER BY stat_time
        """,
        params=params,
        engine=engine,
    )
    hourly = build_hourly_timeseries(stats)
    forecast = forecast_next_hours(hourly, steps=3)
    if hourly.empty:
        st.info("暂无小时统计数据。")
    else:
        st.info(trend_summary(hourly, forecast))
        st.plotly_chart(make_hourly_trend_chart(hourly, forecast), use_container_width=True)
        detail = hourly[["stat_time", "开始订单数量", "结束订单数量", "总订单数量", "3小时移动均值", "时段类型"]].tail(24)
        st.dataframe(detail, use_container_width=True, hide_index=True)
