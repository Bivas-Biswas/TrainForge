from __future__ import annotations

import sqlite3
from typing import Any

from core.config import DB_PATH


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS models(
    token TEXT PRIMARY KEY,
    status TEXT,
    path TEXT,
    model_type TEXT DEFAULT 'svm',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_inference_at TIMESTAMP,
    model_size_bytes INTEGER
)
"""


def _get_column_type(cursor: sqlite3.Cursor, table: str, column: str) -> str | None:
    cursor.execute(f"PRAGMA table_info({table})")
    for _, name, col_type, *_ in cursor.fetchall():
        if name == column:
            return (col_type or "").upper()
    return None


def _migrate_last_inference_to_timestamp(cursor: sqlite3.Cursor) -> None:
    # SQLite cannot alter a column type in-place; rebuild table to normalize schema.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models_new(
            token TEXT PRIMARY KEY,
            status TEXT,
            path TEXT,
            model_type TEXT DEFAULT 'svm',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_inference_at TIMESTAMP,
            model_size_bytes INTEGER
        )
        """
    )
    cursor.execute(
        """
        INSERT INTO models_new (
            token, status, path, model_type, error_message, created_at, last_inference_at, model_size_bytes
        )
        SELECT
            token,
            status,
            path,
            COALESCE(model_type, 'svm'),
            error_message,
            COALESCE(created_at, CURRENT_TIMESTAMP),
            last_inference_at,
            model_size_bytes
        FROM models
        """
    )
    cursor.execute("DROP TABLE models")
    cursor.execute("ALTER TABLE models_new RENAME TO models")


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

        try:
            c.execute("ALTER TABLE models ADD COLUMN error_message TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            c.execute("ALTER TABLE models ADD COLUMN created_at TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            c.execute("ALTER TABLE models ADD COLUMN last_inference_at TIMESTAMP")
        except sqlite3.OperationalError:
            pass

        try:
            c.execute("ALTER TABLE models ADD COLUMN model_size_bytes INTEGER")
        except sqlite3.OperationalError:
            pass

        if _get_column_type(c, "models", "last_inference_at") == "TEXT":
            _migrate_last_inference_to_timestamp(c)

        # Backfill timestamps for legacy rows that predate these columns.
        c.execute(
            "UPDATE models SET created_at=CURRENT_TIMESTAMP WHERE created_at IS NULL"
        )


def create_model_record(token: str, status: str, path: str, model_type: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO models (
                token, status, path, model_type, error_message, created_at, last_inference_at, model_size_bytes
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, NULL)
            """,
            (token, status, path, model_type, None),
        )


def mark_model_completed(token: str, model_size_bytes: int) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            UPDATE models
            SET status='completed', error_message=NULL, model_size_bytes=?
            WHERE token=?
            """,
            (model_size_bytes, token),
        )


def update_model_status(token: str, status: str, error_message: str | None = None) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE models SET status=?, error_message=? WHERE token=?",
            (status, error_message, token),
        )


def fetch_model_status(token: str) -> str | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT status FROM models WHERE token=?", (token,))
        row = c.fetchone()
    return row[0] if row else None


def fetch_tokens_by_status(status: str) -> list[str]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT token FROM models WHERE status=?", (status,))
        rows = c.fetchall()
    return [row[0] for row in rows]


def fetch_model_status_details(token: str) -> tuple[str, str | None] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT status, error_message FROM models WHERE token=?", (token,))
        row = c.fetchone()

    if not row:
        return None

    return row[0], row[1]


def fetch_model_path_and_status(token: str) -> tuple[str, str] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT path,status FROM models WHERE token=?", (token,))
        row = c.fetchone()

    if not row:
        return None

    return row[0], row[1]


def mark_token_failed(token: str | None, reason: str | None = None) -> None:
    if not token:
        return
    update_model_status(token, "failed", reason)


def touch_last_inference_at(token: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE models SET last_inference_at=CURRENT_TIMESTAMP WHERE token=?",
            (token,),
        )


def fetch_model_record(token: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT token,status,path,model_type,error_message,created_at,last_inference_at,model_size_bytes
            FROM models
            WHERE token=?
            """,
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
        "error_message": row[4],
        "created_at": row[5],
        "last_inference_at": row[6],
        "model_size_bytes": row[7],
    }
