import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, require_login
from modules.database import get_engine, init_db, query_df
from modules.dispatch import dispatch_explanation, dispatch_gap_summary, generate_and_save_dispatch
from modules.geo_context import nearest_area_name
from modules.map_view import dispatch_route_map
from modules.streamlit_map import render_map
from modules.ui import show_page_error


st.set_page_config(page_title="调度优化", layout="wide")


try:
    require_login([ROLE_ADMIN, ROLE_OPERATOR])
    engine = get_engine()
    init_db(engine)

    st.title("调度优化")
    st.caption(
        "系统根据预测供需缺口，将车辆从余车网格区域调往缺车网格区域。"
        "调度方案基于预测结果生成，属于运营区域约束下的网格级启发式调度建议。"
    )

    if st.button("生成调度方案", use_container_width=True):
        with st.spinner("正在根据预测供需缺口生成调度方案..."):
            plans = generate_and_save_dispatch(engine=engine, threshold=0.5)
        if plans.empty:
            st.warning("暂无可生成的调度任务。请先生成预测结果，或等待更多订单数据形成明显余车区与缺车区。")
        else:
            st.success(f"已生成 {len(plans)} 条调度任务。")

    plans = query_df("SELECT * FROM dispatch_plans ORDER BY priority DESC, dispatch_bikes DESC, distance_km ASC", engine=engine)
    summary = dispatch_gap_summary(engine=engine, plans=plans)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("调度前总缺口", summary["before_gap"])
    k2.metric("建议调度车辆数", summary["dispatch_bikes"])
    k3.metric("调度后剩余缺口", summary["after_gap"])
    k4.metric("缺口缓解率", f"{summary['relief_rate'] * 100:.1f}%")
    k5.metric("调度任务数", summary["task_count"])
    k6.metric("总调度距离", f"{summary['total_distance']:.2f} km")

    st.info(dispatch_explanation(summary))

    if plans.empty:
        st.info("暂无调度方案。请先在“热点预测”页面生成预测结果。")
        st.stop()

    st.subheader("调度任务表")
    display = plans.copy()
    display["调出区域"] = display.apply(lambda row: nearest_area_name(row["source_lng"], row["source_lat"])[0], axis=1)
    display["调入区域"] = display.apply(lambda row: nearest_area_name(row["target_lng"], row["target_lat"])[0], axis=1)
    display = display.rename(
        columns={
            "source_grid_id": "调出网格",
            "target_grid_id": "调入网格",
            "dispatch_bikes": "调度车辆数",
            "distance_km": "距离km",
            "priority": "优先级",
            "reason": "调度原因",
        }
    )
    display_columns = ["调出区域", "调入区域", "调出网格", "调入网格", "调度车辆数", "距离km", "优先级", "调度原因"]
    st.dataframe(display[display_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "导出调度方案 CSV",
        display.to_csv(index=False).encode("utf-8-sig"),
        "dispatch_plans.csv",
        "text/csv",
    )

    st.subheader("调度路线地图")
    render_map(dispatch_route_map(plans), height=540, key="dispatch_route_map")

    with st.expander("调度效果测试指标", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("平均调度距离", f"{summary['avg_distance']:.2f} km")
        c2.metric("高优先级任务数", summary["high_priority_tasks"])
        c3.metric("启发式算法", "供需缺口 + 空间距离")
        st.caption("调度前总缺口为所有缺车网格区域的正向供需缺口之和；调度后剩余缺口按建议调度车辆数进行抵扣，用于毕业论文中的调度效果测试。")
except Exception as exc:
    show_page_error(exc, "调度优化页面加载失败，请检查预测结果或调度方案表。")
