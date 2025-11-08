"""
用户管理模块，基于SQLite实现基础的用户注册、登录与资料维护
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from werkzeug.security import check_password_hash, generate_password_hash


class UserManager:
    """封装用户相关的数据库操作"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    full_name TEXT,
                    bio TEXT,
                    avatar_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Optional[str]]:
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "full_name": row["full_name"],
            "bio": row["bio"],
            "avatar_url": row["avatar_url"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        username = username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(password) < 6:
            raise ValueError("密码长度不能少于6位")

        now = datetime.utcnow().isoformat()
        password_hash = generate_password_hash(password)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, email, full_name, bio, avatar_url, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        password_hash,
                        email,
                        full_name,
                        "",
                        "",
                        now,
                        now,
                    ),
                )
                conn.commit()
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("用户名已存在")

        return self.get_user_by_id(user_id)  # type: ignore[arg-type]

    def authenticate_user(self, username: str, password: str) -> Dict[str, Optional[str]]:
        username = username.strip()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if row is None or not check_password_hash(row["password_hash"], password):
            raise ValueError("用户名或密码错误")

        return self._row_to_dict(row)

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Optional[str]]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Optional[str]]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username.strip(),)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def update_user(
        self,
        user_id: int,
        *,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        bio: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE users
                SET email = COALESCE(?, email),
                    full_name = COALESCE(?, full_name),
                    bio = COALESCE(?, bio),
                    avatar_url = COALESCE(?, avatar_url),
                    updated_at = ?
                WHERE id = ?
                """,
                (email, full_name, bio, avatar_url, now, user_id),
            )
            conn.commit()

        user = self.get_user_by_id(user_id)
        if user is None:
            raise ValueError("用户不存在")
        return user

