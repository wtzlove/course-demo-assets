import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import ensure_directories
from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, require_login
from modules.database import get_engine, init_db, query_df, system_summary
from modules.dispatch import dispatch_gap_summary
from modules.geo_context import nearest_area_name
from modules.map_view import hotspot_map
from modules.streamlit_map import render_map
from modules.ui import show_page_error


st.set_page_config(page_title="共享单车停放热点预测与调度系统", layout="wide")


def normalize_level(value: str) -> str:
    if value in {"高", "中", "低"}:
        return value
    return {"High": "高", "Medium": "中", "Low": "低"}.get(str(value), str(value))


def build_hourly_chart(hourly: pd.DataFrame):
    if hourly.empty:
        return None
    chart_df = hourly.copy()
    chart_df["stat_hour"] = pd.to_datetime(chart_df["stat_hour"], errors="coerce")
    chart_df = chart_df.dropna(subset=["stat_hour"]).sort_values("stat_hour")
    fig = px.line(
        chart_df,
        x="stat_hour",
        y="orders",
        markers=True,
        text="orders",
        title="小时订单趋势",
    )
    fig.update_traces(textposition="top center", line=dict(width=3), marker=dict(size=7))
    fig.update_layout(
        height=380,
        xaxis_title="时间",
        yaxis_title="订单数量（单）",
        margin=dict(l=40, r=30, t=70, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%m-%d %H:%M", tickangle=-35)
    return fig


def build_priority_chart(priority: pd.DataFrame):
    if priority.empty:
        return None
    df = priority.copy()
    df["priority"] = df["priority"].map(normalize_level)
    order = ["高", "中", "低"]
    df["priority"] = pd.Categorical(df["priority"], categories=order, ordered=True)
    df = df.sort_values("priority")
    fig = px.bar(
        df,
        x="priority",
        y="tasks",
        text="tasks",
        color="priority",
        title="调度任务优先级分布",
        color_discrete_map={"高": "#ef4444", "中": "#f59e0b", "低": "#22c55e"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=260,
        showlegend=False,
        xaxis_title="优先级",
        yaxis_title="任务数量（条）",
        margin=dict(l=35, r=35, t=60, b=45),
    )
    return fig


try:
    ensure_directories()
    engine = get_engine()
    init_db(engine)
    user = require_login([ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER])

    st.title("基于开放接口的共享单车停放热点准实时预测与调度优化系统")
    st.caption(
        f"梁山县哈啰共享单车订单数据驾驶舱 · 当前角色：{user['role']} · "
        "增量采集 / 热点识别 / 短时预测 / 调度优化 / GIS 可视化 / AI 辅助分析"
    )

    summary = system_summary(engine)
    st.markdown(
        f"<div class='section-note'>数据更新时间：{summary['last_crawl'] or '暂无采集记录'}；"
        f"最新模型训练时间：{summary['last_model_time'] or '暂无训练记录'}。</div>",
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("累计订单数", summary["clean_orders"])
    metric_cols[1].metric("累计车辆数", summary["bike_count"])
    metric_cols[2].metric("高热点区域数", summary["hotspot_high"])
    metric_cols[3].metric("预测高风险区域数", summary["prediction_high"])
    metric_cols[4].metric("建议调度车辆数", summary["dispatch_bikes"])
    metric_cols[5].metric("最新模型训练", summary["last_model_time"] or "暂无")

    st.divider()
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("订单趋势")
        hourly = query_df(
            """
            SELECT substr(start_time, 1, 13) AS stat_hour,
                   COUNT(*) AS orders
            FROM clean_orders
            GROUP BY substr(start_time, 1, 13)
            ORDER BY stat_hour
            """,
            engine=engine,
        )
        chart = build_hourly_chart(hourly)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("暂无订单趋势数据。请先在“数据采集”页面采集接口数据。")

        st.subheader("最近调度建议")
        plans = query_df("SELECT * FROM dispatch_plans ORDER BY id DESC LIMIT 8", engine=engine)
        if plans.empty:
            st.info("暂无调度方案。请先在“热点预测”页面生成预测结果，再进入“调度优化”页面生成方案。")
        else:
            gap = dispatch_gap_summary(engine=engine, plans=plans)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("调度任务数", gap["task_count"])
            c2.metric("建议调度车辆", gap["dispatch_bikes"])
            c3.metric("缺口缓解率", f"{gap['relief_rate'] * 100:.1f}%")
            c4.metric("总调度距离", f"{gap['total_distance']:.2f} km")

            for row in plans.head(4).itertuples():
                source_area = nearest_area_name(row.source_lng, row.source_lat)[0]
                target_area = nearest_area_name(row.target_lng, row.target_lat)[0]
                priority = normalize_level(row.priority)
                color = {"高": "#ef4444", "中": "#f59e0b", "低": "#22c55e"}.get(priority, "#0f7bff")
                st.markdown(
                    f"""
                    <div class="dispatch-card" style="border-left-color:{color};">
                      <div class="dispatch-title">{source_area} → {target_area}</div>
                      <div class="dispatch-meta">
                        建议调度 <b>{int(row.dispatch_bikes)}</b> 辆 · 距离 <b>{row.distance_km}</b> km · 优先级 <b>{priority}</b>
                      </div>
                      <div class="dispatch-reason">{row.reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("当前停放热点地图")
        hotspots = query_df("SELECT * FROM hotspot_results ORDER BY hotspot_score DESC LIMIT 80", engine=engine)
        if hotspots.empty:
            st.info("暂无热点识别结果。采集并清洗订单后，可在“热点监测”页面重新计算热点。")
        else:
            render_map(hotspot_map(hotspots), height=470, key="home_hotspot_map")

        priority = query_df(
            """
            SELECT priority, COUNT(*) AS tasks
            FROM dispatch_plans
            GROUP BY priority
            """,
            engine=engine,
        )
        pfig = build_priority_chart(priority)
        if pfig:
            st.plotly_chart(pfig, use_container_width=True)
        else:
            st.info("暂无调度优先级分布。")

    st.divider()
    st.markdown(
        """
本系统采用本地 Web 化方式实现，基于 Streamlit 构建交互式界面，使用 SQLite 数据库存储共享单车订单数据。
系统通过开放接口持续获取梁山县共享单车订单数据，在入库前删除手机号并对用户标识脱敏；基于累积数据完成网格化热点识别、
短时停放压力预测和调度建议生成，适用于本科毕业设计论文截图、答辩演示和本地运行。
"""
    )
except Exception as exc:
    show_page_error(exc, "首页加载失败，请检查数据库是否已初始化或依赖是否安装完整。")
