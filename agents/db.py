"""Shared read-only DuckDB connection used by all tools."""

from pathlib import Path
from threading import Lock

import duckdb

DB_PATH = Path(__file__).resolve().parent.parent / "olist.duckdb"

_lock = Lock()
_con: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    global _con
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found. Run `python db/build_db.py` first."
        )
    with _lock:
        if _con is None:
            _con = duckdb.connect(str(DB_PATH), read_only=True)
    return _con
