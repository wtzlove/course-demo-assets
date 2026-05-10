import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, require_login
from modules.database import get_engine, init_db, query_df
from modules.dispatch import dispatch_gap_summary, generate_and_save_dispatch
from modules.geo_context import nearest_area_name
from modules.map_view import dispatch_map
from modules.streamlit_map import render_map
from modules.ui import show_page_error


st.set_page_config(page_title="调度优化", layout="wide")


def normalize_priority(value: str) -> str:
    if value in {"高", "中", "低"}:
        return value
    return {"High": "高", "Medium": "中", "Low": "低"}.get(str(value), str(value))


try:
    require_login([ROLE_ADMIN, ROLE_OPERATOR])
    engine = get_engine()
    init_db(engine)

    st.title("调度优化")
    st.caption("根据预测的借车需求与停放压力计算供需缺口，采用贪心策略生成车辆调度路径和车辆分配建议。")

    threshold = st.slider("供需差阈值", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    if st.button("生成调度方案", use_container_width=True):
        with st.spinner("正在计算供需缺口并生成调度方案..."):
            plans = generate_and_save_dispatch(engine=engine, threshold=threshold)
        if plans.empty:
            st.warning("暂无可生成的调度任务。请先生成预测结果，或降低供需差阈值。")
        else:
            st.success(f"已生成 {len(plans)} 条调度任务。")

    plans = query_df("SELECT * FROM dispatch_plans ORDER BY dispatch_bikes DESC, distance_km ASC", engine=engine)
    gap = dispatch_gap_summary(engine=engine, plans=plans)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("调度前缺口", gap["before_gap"])
    k2.metric("调度后缺口", gap["after_gap"])
    k3.metric("缺口缓解率", f"{gap['relief_rate'] * 100:.1f}%")
    k4.metric("调度任务数", gap["task_count"])
    k5.metric("建议调度车辆", gap["dispatch_bikes"])
    k6.metric("总调度距离", f"{gap['total_distance']:.2f} km")

    if plans.empty:
        st.info("暂无调度方案。")
    else:
        st.subheader("调度路线地图")
        render_map(dispatch_map(plans), height=540, key="dispatch_map")

        st.subheader("调度任务表")
        display = plans.copy()
        display["调出区域"] = display.apply(lambda row: nearest_area_name(row["source_lng"], row["source_lat"])[0], axis=1)
        display["调入区域"] = display.apply(lambda row: nearest_area_name(row["target_lng"], row["target_lat"])[0], axis=1)
        display["priority"] = display["priority"].map(normalize_priority)
        display = display.rename(
            columns={
                "source_grid_id": "调出网格",
                "target_grid_id": "调入网格",
                "dispatch_bikes": "分配车辆数",
                "distance_km": "路径距离km",
                "priority": "调度优先级",
                "reason": "调度策略",
            }
        )
        display_columns = [
            "调出区域",
            "调入区域",
            "调出网格",
            "调入网格",
            "分配车辆数",
            "路径距离km",
            "调度优先级",
            "调度策略",
        ]
        st.dataframe(display[display_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "导出调度方案 CSV",
            display.to_csv(index=False).encode("utf-8-sig"),
            "dispatch_plans.csv",
            "text/csv",
        )
except Exception as exc:
    show_page_error(exc, "调度优化页面加载失败，请检查预测结果或调度方案表。")
