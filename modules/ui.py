from __future__ import annotations

import streamlit as st


def screenshot_mode() -> bool:
    return bool(st.session_state.get("screenshot_mode", False))


def screenshot_mode_toggle() -> None:
    value = st.toggle(
        "论文截图模式",
        value=screenshot_mode(),
        help="开启后隐藏调试信息、接口原文和错误堆栈，只保留适合论文截图的图表、地图、表格和指标。",
    )
    st.session_state["screenshot_mode"] = value


def show_page_error(error: Exception, message: str = "页面加载失败，请检查数据或稍后重试。") -> None:
    st.error(message)
    if not screenshot_mode():
        st.exception(error)


def info_empty(message: str) -> None:
    st.info(message)


def inject_global_css() -> None:
    st.markdown(
        """
<style>
    :root {
        --brand: #0f7bff;
        --brand-dark: #0f3a75;
        --paper-bg: #f6f8fb;
        --card-bg: #ffffff;
        --line: #e3e8f0;
        --muted: #667085;
        --text: #1f2937;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--paper-bg);
    }
    .block-container {
        max-width: 1520px;
        padding-top: 1.15rem;
        padding-left: clamp(1rem, 2.2vw, 2.6rem);
        padding-right: clamp(1rem, 2.2vw, 2.6rem);
        padding-bottom: 2rem;
    }
    h1 {
        color: #111827;
        letter-spacing: 0;
        font-weight: 800;
    }
    h2, h3 {
        color: #1f2937;
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.78rem 0.9rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 800;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--line);
    }
    hr {
        margin: 1.35rem 0;
        border-color: #dbe3ef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        overflow-x: auto;
        border-bottom: 1px solid #dbe3ef;
    }
    .stTabs [data-baseweb="tab"] {
        white-space: nowrap;
        color: #334155;
        font-weight: 650;
    }
    .dashboard-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .section-note {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .dispatch-card {
        border-left: 5px solid #0f7bff;
        background: #ffffff;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
        border-top: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    .dispatch-title {
        font-weight: 750;
        color: #111827;
        margin-bottom: 4px;
    }
    .dispatch-meta {
        color: #334155;
        font-size: 14px;
    }
    .dispatch-reason {
        color: #64748b;
        font-size: 13px;
        margin-top: 4px;
        line-height: 1.5;
    }
    @media (max-width: 1000px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )
