from datetime import datetime, time, timedelta

import pandas as pd
import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, require_login
from modules.database import get_engine, init_db, query_df
from modules.deep_predictor import torch_available
from modules.environment import ensure_environment_for_date, environment_for_date
from modules.geo_context import add_area_names
from modules.map_view import demand_prediction_map
from modules.predictor import train_and_save_predictions
from modules.streamlit_map import render_map
from modules.ui import show_page_error


st.set_page_config(page_title="热点预测", layout="wide")


def fmt_metric(value, digits: int = 3):
    if value is None or value == "":
        return "暂无"
    try:
        return round(float(value), digits)
    except Exception:
        return value


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series even if SQL accidentally produced duplicate column names."""
    values = df[column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce").fillna(0)


def prediction_summary(pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {
            "area_count": 0,
            "total_start": 0.0,
            "total_end": 0.0,
            "total_gap": 0.0,
            "high_risk": 0,
            "model_name": "暂无",
        }
    df = pred.copy()
    df["predicted_start_count"] = numeric_series(df, "predicted_start_count")
    df["predicted_end_count"] = numeric_series(df, "predicted_end_count")
    df["demand_gap"] = numeric_series(df, "demand_gap")
    return {
        "area_count": int(len(df)),
        "total_start": round(float(df["predicted_start_count"].sum()), 2),
        "total_end": round(float(df["predicted_end_count"].sum()), 2),
        "total_gap": round(float(df["demand_gap"].clip(lower=0).sum()), 2),
        "high_risk": int((df["risk_level"].astype(str) == "高").sum()),
        "model_name": str(df["model_name"].iloc[0]) if "model_name" in df.columns and not df.empty else "暂无",
    }


def latest_model_metrics(engine, latest_metrics: dict | None = None) -> dict:
    latest_metrics = latest_metrics or {}
    latest_log = query_df("SELECT * FROM model_logs ORDER BY id DESC LIMIT 1", engine=engine)
    row = latest_log.iloc[0].to_dict() if not latest_log.empty else {}
    return {
        "model_name": latest_metrics.get("model_name") or row.get("model_name") or "暂无",
        "sample_count": latest_metrics.get("sample_count", row.get("sample_count", 0) or 0),
        "mae": latest_metrics.get("mae", row.get("mae")),
        "rmse": latest_metrics.get("rmse", row.get("rmse")),
        "r2": latest_metrics.get("r2", row.get("r2")),
        "train_end_time": row.get("train_end_time", "暂无"),
        "message": latest_metrics.get("message", ""),
    }


def render_model_runtime_status(engine, latest_metrics: dict | None = None) -> None:
    """Show PyTorch availability and the actual prediction layer used by the system."""
    metrics = latest_model_metrics(engine, latest_metrics)
    model_name = str(metrics["model_name"])
    sample_count = int(metrics["sample_count"] or 0)
    is_torch_ready = torch_available()

    if model_name.startswith("CNN-BiLSTM") and is_torch_ready:
        deep_status = "已启用"
    elif model_name.startswith("CNN-BiLSTM"):
        deep_status = "已有CNN结果"
    elif is_torch_ready:
        deep_status = "已安装未启用"
    else:
        deep_status = "自动回退"

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("PyTorch 状态", "已安装" if is_torch_ready else "未安装")
    s2.metric("深度模型状态", deep_status)
    s3.metric("当前采用模型", model_name)
    s4.metric("有效训练样本", sample_count)

    if not is_torch_ready and model_name.startswith("CNN-BiLSTM"):
        st.info("当前运行环境未识别到 PyTorch，但数据库中保留了之前生成的 CNN-BiLSTM 预测结果；如果重新生成预测，系统会自动回退到轻量预测模型。")
    elif not is_torch_ready:
        st.info("当前环境未安装 PyTorch，系统已自动使用轻量预测模型或历史趋势模型；这是正常回退，不影响预测与调度流程。")
    elif not model_name.startswith("CNN-BiLSTM"):
        st.info("PyTorch 已安装，但系统会根据序列样本和训练状态自动决定是否启用 CNN-BiLSTM；未启用时使用轻量预测模型。")


try:
    require_login([ROLE_ADMIN, ROLE_OPERATOR])
    engine = get_engine()
    init_db(engine)

    st.title("热点预测")
    st.caption(
        "系统基于历史订单、网格时序特征、天气与节假日特征，对运营区域约束下的网格级短时借还需求进行预测。"
        "这里将经纬度网格区域视为虚拟站点，用于识别下一小时可能出现的缺车风险和停放压力。"
    )

    c1, c2 = st.columns(2)
    predict_date = c1.date_input("预测日期", value=(datetime.now() + timedelta(hours=1)).date())
    predict_hour = c2.time_input("预测小时", value=time((datetime.now() + timedelta(hours=1)).hour, 0, 0))
    predict_dt = datetime.combine(predict_date, predict_hour)
    env_result = ensure_environment_for_date(predict_date.strftime("%Y-%m-%d"), engine=engine)
    env = environment_for_date(predict_date.strftime("%Y-%m-%d"), engine=engine)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("日期类型", "节假日" if env["is_holiday"] else ("周末" if env["is_weekend"] else "工作日"))
    e2.metric("预测日均温", f"{env['temperature_mean']} °C")
    e3.metric("预测日降水", f"{env['precipitation_sum']} mm")
    e4.metric("预测日最大风速", f"{env['wind_speed_max']} km/h")
    if env_result.get("weather_error"):
        st.warning("天气接口暂未获取成功，系统已使用默认或已有天气特征继续预测。")

    if st.button("生成预测结果", use_container_width=True):
        with st.spinner("正在进行网格级短时借还需求预测..."):
            metrics, predictions = train_and_save_predictions(engine=engine, predict_time=predict_dt)
        if predictions.empty:
            st.warning("暂无网格小时统计数据，无法生成预测。请先采集订单并完成热点统计。")
        else:
            st.session_state["prediction_metrics"] = metrics
            st.success("预测结果已生成。")

    st.subheader("预测引擎状态")
    render_model_runtime_status(engine, st.session_state.get("prediction_metrics"))

    pred = query_df(
        """
        SELECT id,
               grid_id,
               predict_time,
               predicted_end_count,
               predicted_start_count,
               COALESCE(demand_gap, predicted_start_count - predicted_end_count) AS demand_gap,
               risk_level,
               model_name,
               center_lng,
               center_lat,
               created_at
        FROM prediction_results
        ORDER BY demand_gap DESC, predicted_start_count DESC
        """,
        engine=engine,
    )
    if pred.empty:
        st.info("暂无预测结果。点击“生成预测结果”后，系统会自动选择合适的预测方式。")
        st.stop()

    summary = prediction_summary(pred)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("预测区域数量", summary["area_count"])
    k2.metric("预测总借车需求", summary["total_start"])
    k3.metric("预测总还车供给", summary["total_end"])
    k4.metric("总供需缺口", summary["total_gap"])
    k5.metric("高风险区域数量", summary["high_risk"])
    k6.metric("本次采用模型", summary["model_name"])

    st.subheader("预测地图")
    render_map(demand_prediction_map(pred.head(120)), height=540, key="demand_prediction_map")

    st.subheader("预测结果表")
    display = add_area_names(pred)
    display = display.rename(
        columns={
            "predicted_start_count": "预测借车需求",
            "predicted_end_count": "预测还车供给",
            "demand_gap": "供需缺口",
            "risk_level": "风险等级",
            "predict_time": "预测时间",
        }
    )
    display_columns = [
        "grid_id",
        "区域名称",
        "区域类型",
        "距地标km",
        "预测借车需求",
        "预测还车供给",
        "供需缺口",
        "风险等级",
        "预测时间",
    ]
    st.dataframe(display[display_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "导出预测结果 CSV",
        display.to_csv(index=False).encode("utf-8-sig"),
        "prediction_results.csv",
        "text/csv",
    )

    with st.expander("模型运行摘要", expanded=False):
        metrics = latest_model_metrics(engine, st.session_state.get("prediction_metrics"))
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("模型类型", metrics["model_name"])
        m2.metric("样本数", int(metrics["sample_count"] or 0))
        m3.metric("MAE", fmt_metric(metrics["mae"]))
        m4.metric("RMSE", fmt_metric(metrics["rmse"]))
        m5.metric("R²", fmt_metric(metrics["r2"]))
        m6.metric("训练时间", metrics["train_end_time"])
        if metrics["message"]:
            st.caption(metrics["message"])
except Exception as exc:
    show_page_error(exc, "热点预测页面加载失败，请检查网格统计、天气节假日数据或模型文件。")
