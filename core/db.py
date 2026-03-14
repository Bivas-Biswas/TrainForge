from __future__ import annotations

import sqlite3
from typing import Any

from core.config import DB_PATH


DEFAULT_MAX_MODELS_STORAGE_SUPPORT_BYTES = 10 * 1024 * 1024 # 10MB


CREATE_CLIENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clients(
    client_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_online_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_models_storage_bytes INTEGER NOT NULL DEFAULT 0,
    maximum_models_storage_support_bytes INTEGER NOT NULL
)
"""


CREATE_MODELS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS models(
    token TEXT PRIMARY KEY,
    client_id TEXT,
    status TEXT,
    path TEXT,
    model_type TEXT DEFAULT 'svm',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_inference_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
            client_id TEXT,
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
            token, client_id, status, path, model_type, error_message, created_at, last_inference_at, model_size_bytes
        )
        SELECT
            token,
            client_id,
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
        c.execute(CREATE_CLIENTS_TABLE_SQL)
        c.execute(CREATE_MODELS_TABLE_SQL)

        # Backfill a deterministic owner for legacy rows created before client support.
        _register_client_activity(c, "legacy")

        try:
            c.execute("ALTER TABLE models ADD COLUMN client_id TEXT")
        except sqlite3.OperationalError:
            pass

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
        c.execute("UPDATE models SET client_id='legacy' WHERE client_id IS NULL")

        _recompute_clients_storage(c)


def _register_client_activity(
    cursor: sqlite3.Cursor,
    client_id: str,
    maximum_models_storage_support_bytes: int | None = None,
) -> None:
    max_bytes = (
        maximum_models_storage_support_bytes
        if maximum_models_storage_support_bytes is not None
        else DEFAULT_MAX_MODELS_STORAGE_SUPPORT_BYTES
    )

    cursor.execute(
        """
        INSERT INTO clients (
            client_id,
            created_at,
            last_online_at,
            total_models_storage_bytes,
            maximum_models_storage_support_bytes
        )
        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?)
        ON CONFLICT(client_id) DO UPDATE SET
            last_online_at=CURRENT_TIMESTAMP
        """,
        (client_id, max_bytes),
    )


def register_client_activity(
    client_id: str,
    maximum_models_storage_support_bytes: int | None = None,
) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        _register_client_activity(
            c,
            client_id,
            maximum_models_storage_support_bytes=maximum_models_storage_support_bytes,
        )


def fetch_client_storage_info(client_id: str) -> tuple[int, int] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT total_models_storage_bytes, maximum_models_storage_support_bytes
            FROM clients
            WHERE client_id=?
            """,
            (client_id,),
        )
        row = c.fetchone()

    if not row:
        return None

    return int(row[0]), int(row[1])


def fetch_client_details_with_models(client_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT
                client_id,
                created_at,
                last_online_at,
                total_models_storage_bytes,
                maximum_models_storage_support_bytes
            FROM clients
            WHERE client_id=?
            """,
            (client_id,),
        )
        client_row = c.fetchone()

        if client_row is None:
            return None

        c.execute(
            """
            SELECT
                token,
                status,
                path,
                model_type,
                error_message,
                created_at,
                last_inference_at,
                COALESCE(model_size_bytes, 0)
            FROM models
            WHERE client_id=?
            ORDER BY datetime(created_at) DESC, token DESC
            """,
            (client_id,),
        )
        model_rows = c.fetchall()

    return {
        "client_id": client_row[0],
        "created_at": client_row[1],
        "last_online_at": client_row[2],
        "total_models_storage_bytes": int(client_row[3]),
        "maximum_models_storage_support_bytes": int(client_row[4]),
        "models": [
            {
                "token": row[0],
                "status": row[1],
                "path": row[2],
                "model_type": row[3],
                "error_message": row[4],
                "created_at": row[5],
                "last_inference_at": row[6],
                "model_size_bytes": int(row[7]),
            }
            for row in model_rows
        ],
    }


def is_client_over_storage_limit(client_id: str) -> bool:
    info = fetch_client_storage_info(client_id)
    if info is None:
        return False

    total_models_storage_bytes, maximum_models_storage_support_bytes = info
    return total_models_storage_bytes >= maximum_models_storage_support_bytes


def _recompute_clients_storage(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        UPDATE clients
        SET total_models_storage_bytes = COALESCE(
            (
                SELECT SUM(models.model_size_bytes)
                FROM models
                WHERE models.client_id = clients.client_id
                AND models.model_size_bytes IS NOT NULL
            ),
            0
        )
        """
    )


def create_model_record(
    token: str,
    client_id: str,
    status: str,
    path: str,
    model_type: str,
) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        _register_client_activity(c, client_id)
        c.execute(
            """
            INSERT INTO models (
                token, client_id, status, path, model_type, error_message, created_at, last_inference_at, model_size_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, NULL)
            """,
            (token, client_id, status, path, model_type, None),
        )


def mark_model_completed(token: str, model_size_bytes: int) -> tuple[bool, str | None]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT client_id, model_size_bytes FROM models WHERE token=?",
            (token,),
        )
        row = c.fetchone()
        if row is None:
            return False, "unknown token"

        client_id, old_size = row[0], row[1] or 0

        c.execute(
            """
            SELECT total_models_storage_bytes, maximum_models_storage_support_bytes
            FROM clients
            WHERE client_id=?
            """,
            (client_id,),
        )
        client_row = c.fetchone()

        if client_row is None:
            return False, "unknown client"

        current_total = int(client_row[0])
        maximum_supported = int(client_row[1])
        proposed_total = current_total - old_size + model_size_bytes

        if proposed_total > maximum_supported:
            c.execute(
                """
                UPDATE models
                SET status='failed', error_message=?
                WHERE token=?
                """,
                (
                    (
                        "storage quota exceeded: "
                        f"{proposed_total}/{maximum_supported} bytes"
                    ),
                    token,
                ),
            )
            c.execute(
                "UPDATE clients SET last_online_at=CURRENT_TIMESTAMP WHERE client_id=?",
                (client_id,),
            )
            return False, "storage quota exceeded"

        c.execute(
            """
            UPDATE models
            SET status='completed', error_message=NULL, model_size_bytes=?
            WHERE token=?
            """,
            (model_size_bytes, token),
        )

        delta = model_size_bytes - old_size
        c.execute(
            """
            UPDATE clients
            SET
                total_models_storage_bytes=MAX(total_models_storage_bytes + ?, 0),
                last_online_at=CURRENT_TIMESTAMP
            WHERE client_id=?
            """,
            (delta, client_id),
        )

    return True, None


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


def fetch_model_status_details(token: str, client_id: str) -> tuple[str, str | None] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT status, error_message FROM models WHERE token=? AND client_id=?",
            (token, client_id),
        )
        row = c.fetchone()

    if not row:
        return None

    return row[0], row[1]


def fetch_model_path_and_status(token: str, client_id: str) -> tuple[str, str] | None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT path,status FROM models WHERE token=? AND client_id=?",
            (token, client_id),
        )
        row = c.fetchone()

    if not row:
        return None

    return row[0], row[1]


def fetch_expired_models(inactive_ttl_sec: int, limit: int = 100) -> list[dict[str, Any]]:
    threshold_modifier = f"-{inactive_ttl_sec} seconds"

    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT
                token,
                client_id,
                path,
                COALESCE(model_size_bytes, 0),
                status,
                last_inference_at,
                created_at,
                CASE
                    WHEN status='completed' THEN last_inference_at
                    WHEN status='failed' THEN created_at
                    ELSE NULL
                END AS cleanup_reference_at
            FROM models
            WHERE
                (
                    status='completed'
                    AND last_inference_at IS NOT NULL
                    AND datetime(last_inference_at) <= datetime('now', ?)
                )
                OR
                (
                    status='failed'
                    AND created_at IS NOT NULL
                    AND datetime(created_at) <= datetime('now', ?)
                )
            ORDER BY datetime(cleanup_reference_at) ASC
            LIMIT ?
            """,
            (threshold_modifier, threshold_modifier, limit),
        )
        rows = c.fetchall()

    return [
        {
            "token": row[0],
            "client_id": row[1],
            "path": row[2],
            "model_size_bytes": int(row[3] or 0),
            "status": row[4],
            "last_inference_at": row[5],
            "created_at": row[6],
            "cleanup_reference_at": row[7],
        }
        for row in rows
    ]


def delete_model_record(token: str) -> tuple[str | None, bool]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT client_id, path, COALESCE(model_size_bytes, 0) FROM models WHERE token=?",
            (token,),
        )
        row = c.fetchone()

        if row is None:
            return None, False

        client_id, path, model_size_bytes = row[0], row[1], int(row[2] or 0)

        c.execute("DELETE FROM models WHERE token=?", (token,))
        if c.rowcount == 0:
            return None, False

        c.execute(
            """
            UPDATE clients
            SET
                total_models_storage_bytes=MAX(total_models_storage_bytes - ?, 0),
                last_online_at=CURRENT_TIMESTAMP
            WHERE client_id=?
            """,
            (model_size_bytes, client_id),
        )

    return path, True


def delete_client_model_record(client_id: str, token: str) -> tuple[str | None, bool]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT client_id, path, COALESCE(model_size_bytes, 0)
            FROM models
            WHERE token=? AND client_id=?
            """,
            (token, client_id),
        )
        row = c.fetchone()

        if row is None:
            return None, False

        _, path, model_size_bytes = row[0], row[1], int(row[2] or 0)

        c.execute(
            "DELETE FROM models WHERE token=? AND client_id=?",
            (token, client_id),
        )
        if c.rowcount == 0:
            return None, False

        c.execute(
            """
            UPDATE clients
            SET
                total_models_storage_bytes=MAX(total_models_storage_bytes - ?, 0),
                last_online_at=CURRENT_TIMESTAMP
            WHERE client_id=?
            """,
            (model_size_bytes, client_id),
        )

    return path, True


def delete_client_and_models(client_id: str) -> tuple[list[str], bool]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT client_id FROM clients WHERE client_id=?", (client_id,))
        if c.fetchone() is None:
            return [], False

        c.execute("SELECT path FROM models WHERE client_id=?", (client_id,))
        model_paths = [row[0] for row in c.fetchall() if row[0]]

        c.execute("DELETE FROM models WHERE client_id=?", (client_id,))
        c.execute("DELETE FROM clients WHERE client_id=?", (client_id,))

    return model_paths, True


def mark_token_failed(token: str | None, reason: str | None = None) -> None:
    if not token:
        return
    update_model_status(token, "failed", reason)


def touch_last_inference_at(token: str, client_id: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            UPDATE models
            SET last_inference_at=CURRENT_TIMESTAMP
            WHERE token=? AND client_id=?
            """,
            (token, client_id),
        )
        c.execute(
            "UPDATE clients SET last_online_at=CURRENT_TIMESTAMP WHERE client_id=?",
            (client_id,),
        )


def fetch_model_record(token: str, client_id: str | None = None) -> dict[str, Any] | None:
    with get_conn() as conn:
        c = conn.cursor()
        if client_id is None:
            c.execute(
                """
                SELECT token,client_id,status,path,model_type,error_message,created_at,last_inference_at,model_size_bytes
                FROM models
                WHERE token=?
                """,
                (token,),
            )
        else:
            c.execute(
                """
                SELECT token,client_id,status,path,model_type,error_message,created_at,last_inference_at,model_size_bytes
                FROM models
                WHERE token=? AND client_id=?
                """,
                (token, client_id),
            )
        row = c.fetchone()

    if not row:
        return None

    return {
        "token": row[0],
        "client_id": row[1],
        "status": row[2],
        "path": row[3],
        "model_type": row[4],
        "error_message": row[5],
        "created_at": row[6],
        "last_inference_at": row[7],
        "model_size_bytes": row[8],
    }
