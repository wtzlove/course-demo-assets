import streamlit as st

from modules.database import get_engine, init_db
from modules.ui import inject_global_css, screenshot_mode_toggle
from modules.user_manager import authenticate_user, ensure_default_admin


ROLE_ADMIN = "管理员"
ROLE_OPERATOR = "运营人员"
ROLE_VIEWER = "普通用户"


def current_user() -> dict | None:
    return st.session_state.get("user")


def is_logged_in() -> bool:
    return current_user() is not None


def login_form() -> None:
    inject_global_css()
    engine = get_engine()
    init_db(engine)
    ensure_default_admin(engine)
    st.markdown(
        """
<style>
    .login-backdrop {
        min-height: 78vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(circle at 18% 20%, rgba(14, 165, 233, .18), transparent 28%),
            radial-gradient(circle at 84% 12%, rgba(34, 197, 94, .14), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .login-card {
        width: min(520px, 92vw);
        background: rgba(255,255,255,.94);
        border: 1px solid #dbe5ef;
        border-radius: 12px;
        box-shadow: 0 24px 70px rgba(15, 23, 42, .18);
        padding: 2rem;
    }
    .login-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: .35rem;
    }
    .login-subtitle {
        color: #64748b;
        margin-bottom: 1rem;
    }
</style>
<div class="login-backdrop">
  <div class="login-card">
    <div class="login-title">共享单车热点预测与调度系统</div>
    <div class="login-subtitle">请先登录。默认管理员：admin / admin123</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### 登录")
    login_type = st.radio("登录类型", ["管理员登录", "运营人员登录", "普通用户登录"], horizontal=True)
    expected_role = {
        "管理员登录": ROLE_ADMIN,
        "运营人员登录": ROLE_OPERATOR,
        "普通用户登录": ROLE_VIEWER,
    }[login_type]
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录系统", use_container_width=True)
    if submitted:
        user = authenticate_user(username, password, engine=engine)
        if user:
            if user["role"] != expected_role:
                st.error(f"该账号角色为“{user['role']}”，不能从“{login_type}”入口登录。")
                return
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("用户名或密码错误，或该用户已停用。")


def logout_button() -> None:
    user = current_user()
    if not user:
        return
    with st.sidebar:
        screenshot_mode_toggle()
        st.divider()
        st.caption(f"当前用户：{user['username']}（{user['role']}）")
        if st.button("退出登录", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()


def require_login(allowed_roles: list[str] | None = None) -> dict:
    inject_global_css()
    if not is_logged_in():
        login_form()
        st.stop()
    user = current_user()
    logout_button()
    if allowed_roles and user["role"] not in allowed_roles:
        st.warning("当前角色无权访问该页面。请使用具备权限的账号登录。")
        st.stop()
    return user


def role_allows(roles: list[str]) -> bool:
    user = current_user()
    return bool(user and user["role"] in roles)
