from datetime import datetime, timedelta

import plotly.express as px
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, require_login
from modules.database import get_engine, init_db, query_df
from modules.environment import collect_environment_data
from modules.map_view import weather_map
from modules.streamlit_map import render_map


st.set_page_config(page_title="天气节假日", layout="wide")
require_login([ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER])
engine = get_engine()
init_db(engine)

st.title("天气与节假日")
st.caption("天气和节假日会作为预测模型的外部特征。大风、降水和高温会影响骑行意愿，周末和节假日通常会改变出行结构。")

c1, c2, c3 = st.columns([1, 1, 1])
start_date = c1.date_input("开始日期", value=datetime.now().date() - timedelta(days=7))
end_date = c2.date_input("结束日期", value=datetime.now().date() + timedelta(days=7))
c3.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
if c3.button("更新天气与节假日", use_container_width=True):
    with st.spinner("正在获取天气预报/历史天气，并生成节假日日历..."):
        result = collect_environment_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), engine=engine)
    if result["weather_error"]:
        st.warning(f"节假日已更新 {result['holiday_rows']} 天；天气接口失败：{result['weather_error']}")
    else:
        st.session_state["weather_update_message"] = f"已更新天气 {result['weather_rows']} 天，节假日 {result['holiday_rows']} 天。"
        st.rerun()

if st.session_state.get("weather_update_message"):
    st.success(st.session_state.pop("weather_update_message"))

weather = query_df(
    """
    SELECT w.date,
           w.temperature_mean,
           w.precipitation_sum,
           w.wind_speed_max,
           w.weather_code,
           COALESCE(h.is_weekend, 0) AS is_weekend,
           COALESCE(h.is_holiday, 0) AS is_holiday,
           COALESCE(h.holiday_name, '') AS holiday_name
    FROM weather_daily w
    LEFT JOIN holiday_calendar h ON w.date = h.date
    ORDER BY w.date
    """,
    engine=engine,
)

if weather.empty:
    st.info("暂无天气数据。请点击上方按钮更新。")
    st.stop()

date_options = weather["date"].tolist()
selected_date = st.selectbox("地图显示日期", date_options, index=len(date_options) - 1)
map_weather = weather[weather["date"] == selected_date]
latest = map_weather.iloc[-1] if not map_weather.empty else weather.sort_values("date").iloc[-1]
m1, m2, m3, m4 = st.columns(4)
m1.metric("最新日期", latest["date"])
m2.metric("平均气温", f"{latest['temperature_mean']} °C")
m3.metric("降水量", f"{latest['precipitation_sum']} mm")
m4.metric("最大风速", f"{latest['wind_speed_max']} km/h")

map_mode = st.radio("天气地图模式", ["在线综合天气图（Ventusky）", "在线多要素图（Windy备选）", "本地天气指标图层"], horizontal=True)
if map_mode == "在线综合天气图（Ventusky）":
    layer = st.radio(
        "在线图层",
        ["温度", "降水", "风速"],
        index=0,
        horizontal=True,
    )
    layer_map = {
        "温度": "temperature-2m",
        "降水": "rain-3h",
        "风速": "wind-10m",
    }
    ventusky_url = f"https://www.ventusky.com/?p=35.766;116.132;9&l={layer_map[layer]}"
    components.iframe(ventusky_url, height=600, scrolling=False)
    st.caption("在线综合天气图来自 Ventusky，可切换温度、降水、风速图层。若浏览器限制嵌入，可点击下方链接在新页面打开。")
    st.link_button("打开 Ventusky 梁山县天气图", ventusky_url, use_container_width=True)
elif map_mode == "在线多要素图（Windy备选）":
    windy_layer = st.radio("Windy 图层", ["风力", "温度", "降雨"], index=0, horizontal=True)
    overlay_map = {"风力": "wind", "温度": "temp", "降雨": "rain"}
    windy_url = (
        "https://embed.windy.com/embed2.html"
        "?lat=35.766&lon=116.132&detailLat=35.766&detailLon=116.132"
        f"&zoom=9&level=surface&overlay={overlay_map[windy_layer]}&product=ecmwf"
        "&menu=&message=true&marker=true&calendar=now&pressure=true&type=map&location=coordinates"
    )
    components.iframe(windy_url, height=560, scrolling=False)
    st.caption("Windy 作为在线备选图，可切换风力、温度、降雨。在线图层范围由第三方地图控制，不能完全锁定梁山县边界。")
else:
    render_map(weather_map(map_weather if not map_weather.empty else weather), height=560, key=f"weather_map_{selected_date}")

chart_df = weather.rename(
    columns={
        "date": "日期",
        "temperature_mean": "平均气温",
        "precipitation_sum": "降水量",
        "wind_speed_max": "最大风速",
    }
)
fig = px.line(chart_df, x="日期", y=["平均气温", "降水量", "最大风速"], markers=True, title="天气趋势")
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.subheader("天气日历")
display = weather.copy()
display["date_dt"] = pd.to_datetime(display["date"])
display["week_start"] = display["date_dt"] - pd.to_timedelta(display["date_dt"].dt.weekday, unit="D")
display["weekday"] = display["date_dt"].dt.weekday

weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
header_cols = st.columns(7)
for idx, name in enumerate(weekday_names):
    header_cols[idx].markdown(f"**{name}**")

for _, week_df in display.groupby("week_start"):
    cols = st.columns(7)
    by_weekday = {int(row.weekday): row for row in week_df.itertuples()}
    for idx in range(7):
        row = by_weekday.get(idx)
        if row is None:
            cols[idx].markdown("&nbsp;", unsafe_allow_html=True)
            continue
        date_label = row.date_dt.strftime("%m-%d")
        day_type = row.holiday_name if row.is_holiday else ("周末" if row.is_weekend else "工作日")
        temp = row.temperature_mean
        rain = row.precipitation_sum
        wind = row.wind_speed_max
        color = "#e0f2fe"
        if float(temp or 0) >= 30:
            color = "#fee2e2"
        elif float(rain or 0) > 0:
            color = "#dbeafe"
        elif row.is_holiday or row.is_weekend:
            color = "#fef3c7"
        cols[idx].markdown(
            f"""
            <div style="background:{color};border:1px solid #cbd5e1;border-radius:8px;
                        padding:10px;min-height:118px;font-size:13px;">
              <div style="font-weight:700;color:#0f172a;">{date_label}</div>
              <div style="color:#475569;margin:2px 0 6px;">{day_type}</div>
              <div>气温：{temp}°C</div>
              <div>降水：{rain}mm</div>
              <div>风速：{wind}km/h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
