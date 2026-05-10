import streamlit as st

from modules.auth import ROLE_ADMIN, require_login
from modules.database import get_engine, init_db
from modules.user_manager import add_user, ensure_default_admin, list_users, update_user_status


st.set_page_config(page_title="用户管理", layout="wide")
require_login([ROLE_ADMIN])
engine = get_engine()
init_db(engine)
ensure_default_admin(engine)

st.title("用户管理")
st.caption("本模块用于本地毕设演示。用户密码以 SHA256 哈希保存，不保存明文密码。")

with st.form("add_user"):
    c1, c2, c3 = st.columns(3)
    username = c1.text_input("用户名")
    password = c2.text_input("初始密码", type="password")
    role = c3.selectbox("角色", ["管理员", "运营人员", "普通用户"])
    submitted = st.form_submit_button("添加用户", use_container_width=True)
    if submitted:
        if not username.strip() or not password:
            st.warning("用户名和密码不能为空。")
        else:
            add_user(username, password, role, engine=engine)
            st.success("用户已添加。如用户名已存在，系统会自动忽略重复添加。")

users = list_users(engine)
st.subheader("用户列表")
st.dataframe(users, use_container_width=True, hide_index=True)

st.subheader("启用/停用用户")
if users.empty:
    st.info("暂无用户。")
else:
    c1, c2, c3 = st.columns([1, 1, 1])
    selected_id = c1.selectbox("用户 ID", users["id"].tolist())
    status = c2.selectbox("状态", ["启用", "停用"])
    c3.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
    if c3.button("更新状态", use_container_width=True):
        update_user_status(int(selected_id), status, engine=engine)
        st.success("用户状态已更新。")
        st.rerun()
