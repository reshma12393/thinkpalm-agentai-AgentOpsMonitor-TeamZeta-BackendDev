"""Ensure a modern sqlite3 for ChromaDB on hosts with older system libsqlite3 (< 3.35)."""

from __future__ import annotations

import sys

try:
    import pysqlite3 as _sqlite3  # type: ignore[import-not-found]

    sys.modules["sqlite3"] = _sqlite3
except ImportError:
    pass
