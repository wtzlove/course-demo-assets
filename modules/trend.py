from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PEAK_FACTORS = {
    "早高峰": {"hours": {7, 8, 9}, "factor": 1.12},
    "午间出行": {"hours": {11, 12, 13}, "factor": 1.06},
    "放学/下班": {"hours": {16, 17, 18, 19}, "factor": 1.18},
}


def classify_peak(hour: int) -> str:
    for name, item in PEAK_FACTORS.items():
        if hour in item["hours"]:
            return name
    return "平峰"


def peak_factor(hour: int) -> float:
    for item in PEAK_FACTORS.values():
        if hour in item["hours"]:
            return item["factor"]
    return 1.0


def build_hourly_timeseries(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    df = stats.copy()
    df["stat_time"] = pd.to_datetime(df["stat_time"])
    grouped = (
        df.groupby("stat_time", as_index=False)[["start_count", "end_count"]]
        .sum()
        .sort_values("stat_time")
    )
    grouped["总订单数量"] = grouped["start_count"] + grouped["end_count"]
    grouped["开始订单数量"] = grouped["start_count"]
    grouped["结束订单数量"] = grouped["end_count"]
    grouped["3小时移动均值"] = grouped["总订单数量"].rolling(3, min_periods=1).mean().round(2)
    grouped["时段类型"] = grouped["stat_time"].dt.hour.map(classify_peak)
    return grouped


def forecast_next_hours(hourly: pd.DataFrame, steps: int = 3) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    recent = hourly.tail(min(len(hourly), 12)).copy()
    y = recent["总订单数量"].astype(float).values
    x = np.arange(len(y))
    if len(y) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0, y[-1]
    last_time = pd.to_datetime(hourly["stat_time"].iloc[-1])
    rows = []
    base_recent = float(hourly["总订单数量"].tail(3).mean())
    for step in range(1, steps + 1):
        next_time = last_time + timedelta(hours=step)
        linear_value = intercept + slope * (len(y) + step - 1)
        value = max(0, 0.55 * linear_value + 0.45 * base_recent)
        value *= peak_factor(next_time.hour)
        rows.append(
            {
                "stat_time": next_time,
                "预测总订单数量": round(float(value), 1),
                "时段类型": classify_peak(next_time.hour),
            }
        )
    return pd.DataFrame(rows)


def trend_summary(hourly: pd.DataFrame, forecast: pd.DataFrame) -> str:
    if hourly.empty:
        return "暂无小时趋势数据。"
    recent = hourly.tail(min(6, len(hourly)))
    first = float(recent["总订单数量"].iloc[0])
    last = float(recent["总订单数量"].iloc[-1])
    direction = "上升" if last > first else ("下降" if last < first else "基本平稳")
    next_text = ""
    if not forecast.empty:
        row = forecast.iloc[0]
        next_text = f" 下一小时预测总订单约 {row['预测总订单数量']} 单，时段类型为{row['时段类型']}。"
    return f"最近 {len(recent)} 个统计小时订单趋势为{direction}。{next_text}"


def make_hourly_trend_chart(hourly: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly["stat_time"],
            y=hourly["开始订单数量"],
            mode="lines+markers+text",
            name="开始订单数量",
            text=hourly["开始订单数量"],
            textposition="top center",
            line=dict(color="#60a5fa", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["stat_time"],
            y=hourly["结束订单数量"],
            mode="lines+markers+text",
            name="结束订单数量",
            text=hourly["结束订单数量"],
            textposition="bottom center",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["stat_time"],
            y=hourly["3小时移动均值"],
            mode="lines",
            name="3小时移动均值",
            line=dict(color="#f97316", width=3),
        )
    )
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["stat_time"],
                y=forecast["预测总订单数量"],
                mode="lines+markers+text",
                name="未来趋势预测",
                text=forecast["预测总订单数量"],
                textposition="top center",
                line=dict(color="#dc2626", width=3, dash="dash"),
            )
        )
    fig.update_layout(
        title="小时订单趋势与未来短时预测",
        xaxis_title="统计时间",
        yaxis_title="订单数量",
        legend_title_text="",
        hovermode="x unified",
        height=560,
        annotations=[
            dict(
                text="趋势预测已加入早高峰、午间、放学/下班时段修正",
                xref="paper",
                yref="paper",
                x=0,
                y=1.08,
                showarrow=False,
                font=dict(color="#475569", size=13),
            )
        ],
    )
    return fig
