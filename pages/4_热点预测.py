from datetime import datetime, time, timedelta

import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, require_login
from modules.database import get_engine, init_db, query_df
from modules.environment import ensure_environment_for_date, environment_for_date
from modules.geo_context import add_area_names
from modules.map_view import prediction_map
from modules.predictor import top_k_hotspot_hit_rate, train_and_save_predictions
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


try:
    require_login([ROLE_ADMIN, ROLE_OPERATOR])
    engine = get_engine()
    init_db(engine)

    st.title("热点预测")
    st.caption("以网格小时为样本，融合历史订单、天气和节假日特征，预测下一小时停放压力和借车需求。")

    c1, c2, c3 = st.columns([1, 1, 1])
    predict_date = c1.date_input("预测日期", value=(datetime.now() + timedelta(hours=1)).date())
    predict_time = c2.time_input("预测小时", value=time((datetime.now() + timedelta(hours=1)).hour, 0, 0))
    predict_dt = datetime.combine(predict_date, predict_time)

    env_result = ensure_environment_for_date(predict_date.strftime("%Y-%m-%d"), engine=engine)
    env = environment_for_date(predict_date.strftime("%Y-%m-%d"), engine=engine)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("预测日均温", f"{env['temperature_mean']} °C")
    e2.metric("预测日降水", f"{env['precipitation_sum']} mm")
    e3.metric("预测日最大风速", f"{env['wind_speed_max']} km/h")
    e4.metric("日期类型", "节假日" if env["is_holiday"] else ("周末" if env["is_weekend"] else "工作日"))
    if env_result.get("weather_error"):
        st.warning("预测日期天气暂未获取成功，模型会使用默认天气特征；可到“天气节假日”页面更新后再训练。")

    if c3.button("重新训练模型并预测", use_container_width=True):
        with st.spinner("正在训练模型并生成预测结果..."):
            metrics, predictions = train_and_save_predictions(engine=engine, predict_time=predict_dt)
        if predictions.empty:
            st.warning("暂无网格小时统计数据，无法训练模型。请先采集并清洗订单。")
        else:
            hit_rate = top_k_hotspot_hit_rate(engine=engine, k=10)
            st.success("预测完成")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("模型", metrics.get("model_name") or "暂无")
            m2.metric("训练样本数", metrics.get("sample_count", 0))
            m3.metric("MAE", fmt_metric(metrics.get("mae")))
            m4.metric("RMSE", fmt_metric(metrics.get("rmse")))
            m5.metric("R²", fmt_metric(metrics.get("r2")))
            m6.metric("Top-K 命中率", "暂无" if hit_rate is None else f"{hit_rate * 100:.1f}%")
            if metrics.get("sample_count", 0) < 30:
                st.info("当前数据量较少，模型评价仅用于系统流程验证；随着历史数据积累，评价会更稳定。")

    st.subheader("最新模型评价")
    latest_log = query_df("SELECT * FROM model_logs ORDER BY id DESC LIMIT 1", engine=engine)
    hit_rate = top_k_hotspot_hit_rate(engine=engine, k=10)
    if latest_log.empty:
        st.info("暂无模型训练日志。")
    else:
        row = latest_log.iloc[0]
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("模型名称", row.get("model_name") or "暂无")
        l2.metric("样本数", int(row.get("sample_count") or 0))
        l3.metric("MAE", fmt_metric(row.get("mae")))
        l4.metric("RMSE", fmt_metric(row.get("rmse")))
        l5.metric("R²", fmt_metric(row.get("r2")))
        l6.metric("Top-K 热点命中率", "暂无" if hit_rate is None else f"{hit_rate * 100:.1f}%")

    pred = query_df("SELECT * FROM prediction_results ORDER BY predicted_end_count DESC", engine=engine)
    if pred.empty:
        st.info("暂无预测结果。请点击“重新训练模型并预测”。")
    else:
        st.subheader("预测高风险热点地图")
        render_map(prediction_map(pred.head(80)), height=540, key="prediction_map")

        pred_display = add_area_names(pred)
        pred_display["供需缺口"] = (
            pred_display["predicted_start_count"].astype(float) - pred_display["predicted_end_count"].astype(float)
        ).round(2)
        pred_display = pred_display.rename(
            columns={
                "grid_id": "网格编号",
                "predict_time": "预测时间",
                "predicted_end_count": "预测结束订单数量",
                "predicted_start_count": "预测开始订单数量",
                "risk_level": "风险等级",
                "model_name": "模型名称",
            }
        )
        display_columns = [
            "区域名称",
            "区域类型",
            "距地标km",
            "网格编号",
            "预测时间",
            "预测开始订单数量",
            "预测结束订单数量",
            "供需缺口",
            "风险等级",
            "模型名称",
        ]
        st.dataframe(pred_display[display_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "导出预测结果 CSV",
            pred_display.to_csv(index=False).encode("utf-8-sig"),
            "prediction_results.csv",
            "text/csv",
        )

    st.subheader("模型训练日志")
    logs = query_df("SELECT * FROM model_logs ORDER BY id DESC LIMIT 20", engine=engine)
    if logs.empty:
        st.info("暂无模型训练日志。")
    else:
        st.dataframe(logs, use_container_width=True, hide_index=True)
except Exception as exc:
    show_page_error(exc, "热点预测页面加载失败，请检查网格统计、天气节假日数据或模型文件。")
