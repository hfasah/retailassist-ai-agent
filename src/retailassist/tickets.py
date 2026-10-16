"""Support tickets for requests the agent can't handle itself, stored in SQLite."""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from .store import DATA_DIR

TicketCategory = Literal[
    "return", "refund", "cancellation", "address_change", "damaged_item", "missing_package", "other"
]
TicketStatus = Literal["open", "in_progress", "closed"]

CATEGORIES = get_args(TicketCategory)
STATUSES = get_args(TicketStatus)


class TicketStore:
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                order_id TEXT,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL
            )"""
        )
        self._db.commit()

    def create(self, *, customer_email: str, category: str, message: str,
               order_id: str | None = None, source: str = "api") -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO tickets (created_at, updated_at, customer_email, order_id, category, message, status, source)"
                " VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
                (now, now, customer_email.strip().lower(), order_id, category, message, source),
            )
            self._db.commit()
            return self._get(cursor.lastrowid)

    def get(self, ticket_id: int) -> dict | None:
        with self._lock:
            return self._get(ticket_id)

    def list(self, status: str | None = None) -> list[dict]:
        with self._lock:
            if status:
                rows = self._db.execute("SELECT * FROM tickets WHERE status = ? ORDER BY id", (status,))
            else:
                rows = self._db.execute("SELECT * FROM tickets ORDER BY id")
            return [_to_dict(row) for row in rows]

    def update_status(self, ticket_id: int, status: str) -> dict | None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._db.execute("UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?", (status, now, ticket_id))
            self._db.commit()
            return self._get(ticket_id)

    def _get(self, ticket_id: int) -> dict | None:
        row = self._db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return _to_dict(row) if row else None


def _to_dict(row: sqlite3.Row) -> dict:
    ticket = dict(row)
    ticket["ticket_id"] = f"TCK-{ticket['id']:04d}"
    return ticket


ticket_store = TicketStore(os.environ.get("RETAILASSIST_TICKETS_DB", DATA_DIR / "tickets.db"))
