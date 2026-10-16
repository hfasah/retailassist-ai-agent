"""Human approval for write actions, stored in SQLite.

Anything that changes data on a customer's behalf, opening a support ticket or submitting a refund
request, is first saved here as a pending approval. A staff member approves or rejects it through the
API (/approvals), and only then does it run:

- From the chat agent, the LangGraph graph pauses at its approval_gate node and resumes once the
  request is decided (see graph.py).
- From the MCP server, the approval endpoint runs the action itself.

Callers must check order access (access.py) before proposing an action; run_action checks again.
"""

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .refunds import RefundReason, describe_refund, refund_outcome, submit_refund
from .schemas import Details, Email, Message, OrderId, Sku
from .store import DATA_DIR, find_order
from .tickets import TicketCategory, open_ticket

Action = Literal["create_ticket", "request_refund"]
ApprovalStatus = Literal["pending", "approved", "rejected"]


class TicketArgs(BaseModel):
    email: Email
    category: TicketCategory
    message: Message
    order_id: OrderId | None = None


class RefundArgs(BaseModel):
    order_id: OrderId
    email: Email
    reason: RefundReason
    skus: list[Sku] | None = Field(None, max_length=20)
    details: Details = ""


@dataclass
class Proposal:
    """A write action waiting for a person to approve it."""
    action: Action
    args: dict
    title: str          # one line, shown to the customer and to staff
    details: str        # exactly what will happen, for staff
    customer_email: str
    order_id: str | None


class ApprovalStore:
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                decided_at TEXT,
                source TEXT NOT NULL,
                thread_id TEXT,
                call_id TEXT,
                action TEXT NOT NULL,
                args TEXT NOT NULL,
                title TEXT NOT NULL,
                details TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                order_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                note TEXT NOT NULL DEFAULT '',
                result TEXT,
                UNIQUE (thread_id, call_id)
            )"""
        )
        self._db.commit()

    def submit(self, proposal: Proposal, *, source: str, thread_id: str | None = None,
               call_id: str | None = None) -> dict:
        """Save a pending approval. For a chat tool call, submitting it again returns the saved one."""
        with self._lock:
            if thread_id and call_id:
                row = self._db.execute("SELECT * FROM approvals WHERE thread_id = ? AND call_id = ?",
                                       (thread_id, call_id)).fetchone()
                if row:
                    return _to_dict(row)
            cursor = self._db.execute(
                "INSERT INTO approvals (created_at, source, thread_id, call_id, action, args, title, details,"
                " customer_email, order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), source, thread_id, call_id, proposal.action, json.dumps(proposal.args), proposal.title,
                 proposal.details, proposal.customer_email, proposal.order_id))
            self._db.commit()
            return self._get(cursor.lastrowid)

    def get(self, approval_id: int) -> dict | None:
        with self._lock:
            return self._get(approval_id)

    def list(self, status: str | None = None, thread_id: str | None = None) -> list[dict]:
        query, params = "SELECT * FROM approvals WHERE 1 = 1", []
        if status:
            query, params = query + " AND status = ?", params + [status]
        if thread_id:
            query, params = query + " AND thread_id = ?", params + [thread_id]
        with self._lock:
            return [_to_dict(row) for row in self._db.execute(query + " ORDER BY id", params)]

    def find_pending(self, action: Action, order_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM approvals WHERE action = ? AND order_id = ? AND status = 'pending'"
                                   " ORDER BY id LIMIT 1", (action, order_id)).fetchone()
            return _to_dict(row) if row else None

    def decide(self, approval_id: int, status: Literal["approved", "rejected"], note: str = "") -> dict | None:
        """Approve or reject a pending request. Returns None if it isn't pending (already decided or unknown)."""
        with self._lock:
            cursor = self._db.execute("UPDATE approvals SET status = ?, note = ?, decided_at = ? "
                                      "WHERE id = ? AND status = 'pending'", (status, note, _now(), approval_id))
            self._db.commit()
            return self._get(approval_id) if cursor.rowcount else None

    def record_result(self, approval_id: int, result: dict) -> None:
        with self._lock:
            self._db.execute("UPDATE approvals SET result = ? WHERE id = ?", (json.dumps(result), approval_id))
            self._db.commit()

    def _get(self, approval_id: int) -> dict | None:
        row = self._db.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        return _to_dict(row) if row else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_dict(row: sqlite3.Row) -> dict:
    approval = dict(row)
    approval["approval_id"] = f"APR-{approval['id']:04d}"
    approval["args"] = json.loads(approval["args"])
    approval["result"] = json.loads(approval["result"]) if approval["result"] else None
    return approval


approval_store = ApprovalStore(os.environ.get("RETAILASSIST_APPROVALS_DB", DATA_DIR / "approvals.db"))


# --- Proposing and running actions ----------------------------------------------------------------

def propose_ticket(args: TicketArgs, order: dict | None = None) -> Proposal:
    """`order` is the already-authorized order, if the ticket is about one."""
    order_id = order["order_id"] if order else None
    about = f" for {order_id}" if order_id else ""
    kind = args.category.replace("_", " ")
    return Proposal(action="create_ticket", args=args.model_dump() | {"order_id": order_id},
                    title=f"Open {'an' if kind[0] in 'aeiou' else 'a'} {kind} ticket{about}",
                    details=args.message, customer_email=args.email.strip().lower(), order_id=order_id)


def propose_refund(args: RefundArgs, order: dict) -> tuple[dict, Proposal | None]:
    """Check the refund policy for an already-authorized order.

    Returns the policy outcome, plus a proposal if the request qualifies and would open a refund ticket.
    No proposal means nothing would be written (not eligible, or a refund is already open).
    """
    outcome = refund_outcome(order, args.reason, args.skus)
    if not outcome["eligible"] or outcome["already_requested"]:
        return outcome, None
    proposal = Proposal(action="request_refund", args=args.model_dump() | {"order_id": order["order_id"]},
                        title=f"Refund ${outcome['estimated_refund']:.2f} on {order['order_id']} "
                              f"({args.reason.replace('_', ' ')})",
                        details=describe_refund(args.reason, outcome, args.details),
                        customer_email=args.email.strip().lower(), order_id=order["order_id"])
    return outcome, proposal


def run_action(approval: dict, store: ApprovalStore | None = None) -> dict:
    """Run an approved action and record what it did on the approval."""
    if approval["status"] != "approved":
        raise PermissionError(f"{approval['approval_id']} is {approval['status']}, not approved.")
    store = store or approval_store
    args = approval["args"]
    note = f"Approved as {approval['approval_id']}" + (f": {approval['note']}" if approval["note"] else ".")
    if approval["action"] == "create_ticket":
        ticket = open_ticket(email=args["email"], category=args["category"], message=f"{args['message']}\n\n{note}",
                             order_id=args["order_id"], source=approval["source"])
        result = {"ticket_id": ticket["ticket_id"], "status": ticket["status"]}
    elif approval["action"] == "request_refund":
        order = find_order(args["order_id"], args["email"])
        if order is None:  # can't happen for an approval made through the proper checks
            raise PermissionError("The order no longer matches the customer's email.")
        result = submit_refund(order, args["email"], args["reason"], args["skus"], args["details"],
                               source=approval["source"], note=note)
    else:
        raise ValueError(f"Unknown action: {approval['action']}")
    store.record_result(approval["id"], result)
    return result


def public_view(approval: dict) -> dict:
    """What a customer may see about their own request."""
    return {"request_id": approval["approval_id"], "title": approval["title"], "status": approval["status"],
            "note": approval["note"], "result": approval["result"]}

