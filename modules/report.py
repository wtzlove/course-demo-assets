import numpy as np
import pandas as pd
import plotly.express as px

from modules.database import query_df, system_summary
from modules.geo_context import nearest_area_name


def get_daily_orders(engine=None) -> pd.DataFrame:
    return query_df(
        "SELECT start_date AS date, COUNT(*) AS orders FROM clean_orders GROUP BY start_date ORDER BY start_date",
        engine=engine,
    )


def get_hourly_orders(engine=None) -> pd.DataFrame:
    return query_df(
        """
        SELECT substr(start_time, 1, 13) AS stat_hour,
               COUNT(*) AS orders
        FROM clean_orders
        GROUP BY substr(start_time, 1, 13)
        ORDER BY stat_hour
        """,
        engine=engine,
    )


def add_area_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    areas = result.apply(lambda row: nearest_area_name(row["center_lng"], row["center_lat"]), axis=1)
    result["区域名称"] = areas.map(lambda item: item[0])
    result["区域类型"] = areas.map(lambda item: item[2])
    result["距参考地标km"] = areas.map(lambda item: item[1])
    result["依据"] = result.apply(
        lambda row: f"按网格内订单平均经纬度（{row['center_lng']:.5f}, {row['center_lat']:.5f}）匹配最近地标",
        axis=1,
    )
    return result[["区域名称", "区域类型", "距参考地标km", "orders", "grid_id", "依据"]]


def get_top_start_grids(engine=None, limit: int = 10) -> pd.DataFrame:
    df = query_df(
        f"""
        SELECT start_grid_id AS grid_id,
               COUNT(*) AS orders,
               AVG(start_lng) AS center_lng,
               AVG(start_lat) AS center_lat
        FROM clean_orders
        GROUP BY start_grid_id
        ORDER BY orders DESC
        LIMIT {limit}
        """,
        engine=engine,
    )
    return add_area_context(df)


def get_top_end_grids(engine=None, limit: int = 10) -> pd.DataFrame:
    df = query_df(
        f"""
        SELECT end_grid_id AS grid_id,
               COUNT(*) AS orders,
               AVG(end_lng) AS center_lng,
               AVG(end_lat) AS center_lat
        FROM clean_orders
        GROUP BY end_grid_id
        ORDER BY orders DESC
        LIMIT {limit}
        """,
        engine=engine,
    )
    return add_area_context(df)


def get_bike_frequency(engine=None, limit: int = 10) -> pd.DataFrame:
    df = query_df(
        f"""
        SELECT CAST(bike_id AS TEXT) AS bike_id,
               COUNT(*) AS orders
        FROM clean_orders
        WHERE bike_id IS NOT NULL AND bike_id != ''
        GROUP BY bike_id
        ORDER BY orders DESC
        LIMIT {limit}
        """,
        engine=engine,
    )
    if not df.empty:
        df["bike_id"] = df["bike_id"].astype(str)
        df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0).astype(int)
    return df


def make_maintenance_warning_chart(df: pd.DataFrame, threshold: int = 50):
    if df.empty:
        return None
    chart_df = df.copy().rename(columns={"bike_id": "车辆编号", "orders": "订单数量"})
    chart_df["车辆编号"] = chart_df["车辆编号"].astype(str)
    chart_df["订单数量"] = pd.to_numeric(chart_df["订单数量"], errors="coerce").fillna(0).astype(int)
    chart_df = chart_df.sort_values("订单数量", ascending=False).head(10)
    chart_df["预警状态"] = np.where(chart_df["订单数量"] >= threshold, "建议检查", "正常")
    chart_df = chart_df.sort_values("订单数量", ascending=True)
    fig = px.bar(
        chart_df,
        x="订单数量",
        y="车辆编号",
        color="预警状态",
        orientation="h",
        text="订单数量",
        title="车辆维修预警 Top 10",
        color_discrete_map={"建议检查": "#ef4444", "正常": "#8dd3c7"},
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#d95f45", annotation_text=f"维修阈值 {threshold} 次")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="累计订单数量（单）",
        yaxis_title="车辆编号",
        yaxis=dict(type="category", categoryorder="array", categoryarray=chart_df["车辆编号"].tolist()),
        legend_title_text="",
        height=520,
        margin=dict(l=135, r=90, t=70, b=45),
    )
    return fig


def get_ride_metrics(engine=None) -> pd.DataFrame:
    df = query_df(
        """
        SELECT start_lng, start_lat, end_lng, end_lat, ride_time
        FROM clean_orders
        WHERE start_lng IS NOT NULL AND start_lat IS NOT NULL
          AND end_lng IS NOT NULL AND end_lat IS NOT NULL
        """,
        engine=engine,
    )
    if df.empty:
        return df
    for col in ["start_lng", "start_lat", "end_lng", "end_lat", "ride_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    lng1 = np.radians(df["start_lng"])
    lat1 = np.radians(df["start_lat"])
    lng2 = np.radians(df["end_lng"])
    lat2 = np.radians(df["end_lat"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    df["骑行直线距离km"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    df["骑行时长分钟"] = df["ride_time"]
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["骑行直线距离km", "骑行时长分钟"])


def make_daily_chart(df: pd.DataFrame):
    if df.empty:
        return None
    fig = px.bar(df, x="date", y="orders", text="orders", title="每日订单量")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="日期", yaxis_title="订单数量（单）", height=390, margin=dict(l=40, r=35, t=70, b=55))
    return fig


def make_hourly_chart(df: pd.DataFrame):
    if df.empty:
        return None
    chart_df = df.copy()
    chart_df["stat_hour"] = pd.to_datetime(chart_df["stat_hour"], errors="coerce")
    chart_df = chart_df.dropna(subset=["stat_hour"]).sort_values("stat_hour")
    fig = px.line(chart_df, x="stat_hour", y="orders", markers=True, text="orders", title="小时订单趋势")
    fig.update_traces(textposition="top center", line=dict(width=3), marker=dict(size=7))
    fig.update_layout(
        xaxis_title="小时",
        yaxis_title="订单数量（单）",
        height=390,
        margin=dict(l=40, r=35, t=70, b=70),
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%m-%d %H:%M", tickangle=-35)
    return fig


def make_distribution_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    bins: list[float] | None = None,
    suffix: str = "",
):
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    if bins is None:
        upper = values.quantile(0.98)
        bins = np.linspace(values.min(), upper, 9).tolist()
        bins = sorted(set([0] + [round(float(x), 2) for x in bins] + [float("inf")]))

    cut = pd.cut(values, bins=bins, include_lowest=True, right=False)
    grouped = cut.value_counts().sort_index().reset_index()
    grouped.columns = ["interval", "订单数量"]

    def label_interval(interval):
        left = interval.left
        right = interval.right
        if np.isinf(right):
            return f"{left:g}{suffix}以上"
        return f"{left:g}-{right:g}{suffix}"

    grouped["区间"] = grouped["interval"].map(label_interval)
    grouped = grouped.sort_values("订单数量", ascending=True)
    fig = px.bar(grouped, y="区间", x="订单数量", orientation="h", text="订单数量", title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="订单数量（单）",
        yaxis_title="",
        height=430,
        margin=dict(l=90, r=70, t=70, b=45),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=grouped["区间"].tolist())
    return fig


def generate_system_summary_text(engine=None) -> str:
    s = system_summary(engine)
    return (
        f"系统当前累计原始订单 {s['raw_orders']} 条，清洗后有效订单 {s['clean_orders']} 条，"
        f"涉及车辆 {s['bike_count']} 辆。当前识别高等级停放热点 {s['hotspot_high']} 个，"
        f"预测高风险区域 {s['prediction_high']} 个，调度方案建议调动车辆 {s['dispatch_bikes']} 辆。"
        f"最近一次数据更新时间为 {s['last_crawl'] or '暂无'}，最近一次模型训练时间为 {s['last_model_time'] or '暂无'}。"
    )
