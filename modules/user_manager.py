import hashlib
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from modules.database import get_engine, init_db, query_df


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_default_admin(engine=None) -> None:
    engine = engine or get_engine()
    init_db(engine)
    existing = query_df("SELECT COUNT(*) AS c FROM system_users", engine=engine)
    if not existing.empty and int(existing["c"].iloc[0]) > 0:
        return
    add_user("admin", "admin123", "管理员", engine=engine)


def add_user(username: str, password: str, role: str = "普通用户", engine=None) -> None:
    engine = engine or get_engine()
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO system_users
                (username, password_hash, role, status, created_at, last_login)
                VALUES (:username, :password_hash, :role, '启用', :created_at, '')
                """
            ),
            {
                "username": username.strip(),
                "password_hash": hash_password(password),
                "role": role,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


def authenticate_user(username: str, password: str, engine=None) -> dict | None:
    engine = engine or get_engine()
    ensure_default_admin(engine)
    df = query_df(
        """
        SELECT id, username, role, status
        FROM system_users
        WHERE username=:username AND password_hash=:password_hash
        LIMIT 1
        """,
        params={"username": username.strip(), "password_hash": hash_password(password)},
        engine=engine,
    )
    if df.empty:
        return None
    user = df.iloc[0].to_dict()
    if user.get("status") != "启用":
        return None
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE system_users SET last_login=:last_login WHERE id=:id"),
            {"last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "id": int(user["id"])},
        )
    return user


def update_user_status(user_id: int, status: str, engine=None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE system_users SET status=:status WHERE id=:id"),
            {"status": status, "id": user_id},
        )


def list_users(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    ensure_default_admin(engine)
    return query_df(
        "SELECT id, username, role, status, created_at, last_login FROM system_users ORDER BY id",
        engine=engine,
    )
