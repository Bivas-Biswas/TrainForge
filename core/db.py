from __future__ import annotations

import sqlite3
from typing import Any

from core.config import DB_PATH


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS models(
    token TEXT PRIMARY KEY,
    status TEXT,
    path TEXT,
    model_type TEXT DEFAULT 'svm'
)
"""


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(CREATE_TABLE_SQL)

        # Backward-compatible migration for older DB files.
        try:
            c.execute("ALTER TABLE models ADD COLUMN model_type TEXT DEFAULT 'svm'")
        except sqlite3.OperationalError:
            pass


def create_model_record(token: str, status: str, path: str, model_type: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO models VALUES (?, ?, ?, ?)",
            (token, status, path, model_type),
        )


def update_model_status(token: str, status: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE models SET status=? WHERE token=?",
            (status, token),
        )


def fetch_model_status(token: str) -> str | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT status FROM models WHERE token=?", (token,))
        row = c.fetchone()
    return row[0] if row else None


def fetch_model_path_and_status(token: str) -> tuple[str, str] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT path,status FROM models WHERE token=?", (token,))
        row = c.fetchone()

    if not row:
        return None

    return row[0], row[1]


def mark_token_failed(token: str | None) -> None:
    if not token:
        return
    update_model_status(token, "failed")


def fetch_model_record(token: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT token,status,path,model_type FROM models WHERE token=?",
            (token,),
        )
        row = c.fetchone()

    if not row:
        return None

    return {
        "token": row[0],
        "status": row[1],
        "path": row[2],
        "model_type": row[3],
    }
