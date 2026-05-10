import streamlit as st

from modules.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, require_login
from modules.ai_assistant import answer_question, build_context, get_llm_config
from modules.database import get_engine, init_db


st.set_page_config(page_title="AI助手", layout="wide")
require_login([ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER])
engine = get_engine()
init_db(engine)

st.title("AI 助手")
st.caption("默认使用本地规则助手；配置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL 后可调用外部大模型。只发送聚合指标，不发送手机号、用户标识或原始订单明细。")
llm_config = get_llm_config()
if llm_config.get("api_key") and llm_config.get("base_url") and llm_config.get("model"):
    st.success(f"外部大模型已配置：{llm_config.get('model')} · {llm_config.get('base_url')}")
else:
    st.info("尚未配置外部大模型。可在项目根目录 llm_config.toml 中填写 api_key、base_url、model。")

context = build_context(engine)
c1, c2, c3, c4 = st.columns(4)
c1.metric("有效订单", context["clean_orders"])
c2.metric("高热点", context["hotspot_high"])
c3.metric("预测高风险", context["prediction_high"])
c4.metric("建议调度车辆", context["dispatch_bikes"])

questions = [
    "解释当前停放热点",
    "分析热点排行榜",
    "解释预测结果",
    "解释调度方案",
    "根据调度方案生成执行说明",
    "生成今日运营建议",
    "生成论文分析段落",
    "如何提高预测准确率",
    "当前数据量是否足够",
]

cols = st.columns(4)
for i, q in enumerate(questions):
    if cols[i % 4].button(q, use_container_width=True):
        st.session_state["ai_question"] = q

question = st.text_area("请输入问题", value=st.session_state.get("ai_question", ""), height=120)
st.caption("AI 会自动读取热点排行榜、预测结果、调度方案、天气节假日和模型日志等聚合结果；不会发送手机号、用户ID或原始订单明细。")
if st.button("生成回答", use_container_width=True):
    if not question.strip():
        st.warning("请输入问题。")
    else:
        st.session_state["ai_answer"] = answer_question(question, engine=engine)

if st.session_state.get("ai_answer"):
    st.subheader("AI 回答")
    st.write(st.session_state["ai_answer"])
