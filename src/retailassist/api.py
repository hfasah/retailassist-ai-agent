"""HTTP API for RetailAssist.

Run it with:
    uvicorn retailassist.api:app --reload
"""

import os
import secrets
import threading
import uuid
from collections import OrderedDict
from functools import lru_cache
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .agent import RetailAssistant
from .store import ORDERS, PRODUCTS
from .tickets import TicketCategory, TicketStatus, ticket_store
from .tools import find_order

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
MAX_SESSIONS = 1000

app = FastAPI(title="RetailAssist API", version="0.1.0")


# --- Dependencies ---------------------------------------------------------------------------------

@lru_cache
def _make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def get_client() -> anthropic.Anthropic:
    try:
        return _make_client()
    except anthropic.CredentialsError:
        raise HTTPException(500, "The server's Anthropic credentials couldn't be loaded. Set ANTHROPIC_API_KEY.")


def require_staff_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Staff-only endpoints need the X-API-Key header to match RETAILASSIST_STAFF_API_KEY."""
    expected = os.environ.get("RETAILASSIST_STAFF_API_KEY")
    if not expected:
        raise HTTPException(503, "Staff endpoints are disabled: RETAILASSIST_STAFF_API_KEY is not set.")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Missing or invalid API key.")


# --- Chat sessions --------------------------------------------------------------------------------

class _Session:
    def __init__(self, client: anthropic.Anthropic):
        self.assistant = RetailAssistant(client)
        self.lock = threading.Lock()


_sessions: OrderedDict[str, _Session] = OrderedDict()
_sessions_lock = threading.Lock()


def _get_session(session_id: str | None, client: anthropic.Anthropic) -> tuple[str, _Session]:
    with _sessions_lock:
        if session_id is not None:
            session = _sessions.get(session_id)
            if session is None:
                raise HTTPException(404, "Unknown session_id. Omit it to start a new conversation.")
            _sessions.move_to_end(session_id)
            return session_id, session
        session_id = uuid.uuid4().hex
        _sessions[session_id] = _Session(client)
        if len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)  # drop the least recently used conversation
        return session_id, _sessions[session_id]


# --- Models ---------------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Omit to start a new conversation; reuse the returned value to continue it.")


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class TicketCreate(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN)
    category: TicketCategory
    message: str = Field(min_length=1, max_length=4000)
    order_id: str | None = None


class TicketUpdate(BaseModel):
    status: TicketStatus


class Ticket(BaseModel):
    ticket_id: str
    id: int
    created_at: str
    updated_at: str
    customer_email: str
    order_id: str | None
    category: TicketCategory
    message: str
    status: TicketStatus
    source: str


# --- Endpoints ------------------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "products": len(PRODUCTS), "orders": len(ORDERS)}


@app.post("/chat")
def chat(request: ChatRequest, client: Annotated[anthropic.Anthropic, Depends(get_client)]) -> ChatResponse:
    session_id, session = _get_session(request.session_id, client)
    with session.lock:  # one message at a time per conversation
        try:
            reply = session.assistant.ask(request.message)
        except anthropic.AuthenticationError:
            raise HTTPException(500, "The server's Anthropic API key is invalid.")
        except anthropic.CredentialsError:
            raise HTTPException(500, "The server's Anthropic credentials couldn't be loaded. Set ANTHROPIC_API_KEY.")
        except TypeError as e:
            # The SDK raises TypeError before sending anything when no credentials are configured.
            if "authentication method" not in str(e):
                raise
            raise HTTPException(500, "The server has no Anthropic API key. Set ANTHROPIC_API_KEY.")
        except anthropic.RateLimitError:
            raise HTTPException(503, "The assistant is busy. Please try again in a moment.")
        except (anthropic.APIConnectionError, anthropic.APIStatusError):
            raise HTTPException(502, "The assistant couldn't be reached. Please try again.")
    return ChatResponse(session_id=session_id, reply=reply)


@app.get("/orders/{order_id}")
def get_order(order_id: str, email: Annotated[str, Query(description="Email the order was placed with")]) -> dict:
    order = find_order(order_id, email)
    if order is None:
        raise HTTPException(404, "No order matches that order number and email.")
    return order


@app.post("/tickets", status_code=201)
def create_ticket(ticket: TicketCreate) -> Ticket:
    order_id = None
    if ticket.order_id is not None:
        order = find_order(ticket.order_id, ticket.email)
        if order is None:
            raise HTTPException(422, "No order matches that order number and email.")
        order_id = order["order_id"]
    return Ticket(**ticket_store.create(customer_email=ticket.email, category=ticket.category,
                                        message=ticket.message, order_id=order_id, source="api"))


@app.get("/tickets", dependencies=[Depends(require_staff_key)])
def list_tickets(status: TicketStatus | None = None) -> list[Ticket]:
    return [Ticket(**t) for t in ticket_store.list(status)]


@app.get("/tickets/{ticket_id}", dependencies=[Depends(require_staff_key)])
def get_ticket(ticket_id: int) -> Ticket:
    ticket = ticket_store.get(ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found.")
    return Ticket(**ticket)


@app.patch("/tickets/{ticket_id}", dependencies=[Depends(require_staff_key)])
def update_ticket(ticket_id: int, update: TicketUpdate) -> Ticket:
    ticket = ticket_store.update_status(ticket_id, update.status)
    if ticket is None:
        raise HTTPException(404, "Ticket not found.")
    return Ticket(**ticket)
