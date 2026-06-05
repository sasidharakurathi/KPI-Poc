import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "alerts.db"
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id       TEXT    NOT NULL,
                kpi_name     TEXT    NOT NULL,
                alert_type   TEXT    NOT NULL,
                frame_idx    INTEGER NOT NULL,
                confidence   REAL,
                frame_path   TEXT,
                extra_data   TEXT,
                created_at   TEXT    NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_job ON alerts(job_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_kpi ON alerts(kpi_name)")
        con.commit()


def insert_alert(
    job_id: str,
    kpi_name: str,
    alert_type: str,
    frame_idx: int,
    confidence: float = 1.0,
    frame_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> int:
    with _lock, _connect() as con:
        cur = con.execute(
            """
            INSERT INTO alerts
                (job_id, kpi_name, alert_type, frame_idx, confidence, frame_path, extra_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, kpi_name, alert_type, frame_idx,
                confidence, frame_path,
                json.dumps(extra) if extra else None,
                datetime.utcnow().isoformat(),
            ),
        )
        con.commit()
        return cur.lastrowid


def query_alerts(
    job_id: Optional[str] = None,
    kpi_name: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    clauses, params = [], []
    if job_id:
        clauses.append("job_id = ?")
        params.append(job_id)
    if kpi_name:
        clauses.append("kpi_name = ?")
        params.append(kpi_name)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with _connect() as con:
        rows = con.execute(
            f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()

    return [dict(r) for r in rows]
