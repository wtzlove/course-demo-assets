from datetime import datetime, time, timedelta

import streamlit as st

from config.settings import DEFAULT_INTERVAL_MINUTES
from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, require_login
from modules.database import get_engine, init_db, query_df, system_summary
from modules.environment import collect_environment_data
from modules.scheduler import recent_days_range, start_scheduler, stop_scheduler
from modules.workflow import crawl_and_update


st.set_page_config(page_title="数据采集", layout="wide")
require_login([ROLE_ADMIN, ROLE_OPERATOR])
engine = get_engine()
init_db(engine)

st.title("数据采集")
st.caption("接口参数通过 URL 查询字符串传递，系统按 orderGuid 增量去重，并在入库前完成脱敏。")


def run_crawl(start_text: str, end_text: str, pages: int):
    try:
        with st.spinner("正在采集接口数据并更新数据库，请稍候..."):
            result = crawl_and_update(start_text, end_text, max_pages=pages, engine=engine)
    except Exception as exc:
        st.error(f"采集失败：{exc}")
        st.info("如果 PyCharm 单脚本可以采集，请确认当前 Streamlit 进程使用同一个 Python 环境和同一份 .env。系统已按单脚本方式拼接 URL、使用 %20 编码时间空格，并加入 User-Agent。")
        return

    st.success("采集完成")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("采集页数", result["pages"])
    c2.metric("接口订单数", result["fetched_count"])
    c3.metric("新增订单", result["inserted_count"])
    c4.metric("重复订单", result["duplicate_count"])
    c5.metric("清洗入库", result["clean_inserted_count"])


with st.form("manual_crawl"):
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("开始日期", value=datetime.now().date() - timedelta(days=3))
    start_time = c1.time_input("开始时间", value=time(0, 0, 0))
    end_date = c2.date_input("结束日期", value=datetime.now().date())
    end_time = c2.time_input("结束时间", value=time(23, 59, 59))
    max_pages = c3.number_input("最大页数", min_value=1, max_value=500, value=10, step=1)
    submitted = st.form_submit_button("开始采集", use_container_width=True)
    if submitted:
        start_text = datetime.combine(start_date, start_time).strftime("%Y-%m-%d %H:%M:%S")
        end_text = datetime.combine(end_date, end_time).strftime("%Y-%m-%d %H:%M:%S")
        run_crawl(start_text, end_text, int(max_pages))

q1, q2, q3, q4 = st.columns(4)
if q1.button("采集最近三天", use_container_width=True):
    start_text, end_text = recent_days_range(3)
    run_crawl(start_text, end_text, 10)
if q2.button("采集今日数据", use_container_width=True):
    now = datetime.now()
    start_text = datetime.combine(now.date(), time(0, 0, 0)).strftime("%Y-%m-%d %H:%M:%S")
    end_text = now.strftime("%Y-%m-%d %H:%M:%S")
    run_crawl(start_text, end_text, 10)
if q3.button("启动定时采集", use_container_width=True):
    def job():
        s, e = recent_days_range(3)
        crawl_and_update(s, e, max_pages=10, engine=engine)

    st.session_state["scheduler_status"] = start_scheduler(job, DEFAULT_INTERVAL_MINUTES)
if q4.button("停止定时采集", use_container_width=True):
    st.session_state["scheduler_status"] = stop_scheduler()

if st.session_state.get("scheduler_status"):
    st.info(st.session_state["scheduler_status"])

st.subheader("气象与节假日数据")
env1, env2, env3 = st.columns([1, 1, 1])
env_start = env1.date_input("环境数据开始日期", value=datetime.now().date() - timedelta(days=7), key="env_start")
env_end = env2.date_input("环境数据结束日期", value=datetime.now().date() + timedelta(days=7), key="env_end")
if env3.button("采集气象与节假日数据", use_container_width=True):
    with st.spinner("正在采集气象数据并生成节假日日历..."):
        result = collect_environment_data(env_start.strftime("%Y-%m-%d"), env_end.strftime("%Y-%m-%d"), engine=engine)
    if result["weather_error"]:
        st.warning(f"节假日数据已生成 {result['holiday_rows']} 天；天气接口暂时失败：{result['weather_error']}")
    else:
        st.success(f"已写入节假日日历 {result['holiday_rows']} 天，天气数据 {result['weather_rows']} 天。")

st.subheader("数据库状态")
s = system_summary(engine)
m1, m2, m3 = st.columns(3)
m1.metric("数据库累计订单数", s["clean_orders"])
m2.metric("最近一次更新时间", s["last_crawl"] or "暂无")
m3.metric("累计车辆数", s["bike_count"])

st.subheader("采集日志")
logs = query_df("SELECT * FROM crawl_logs ORDER BY id DESC LIMIT 50", engine=engine)
st.dataframe(logs, use_container_width=True, hide_index=True)
