"""HTTP API for RetailAssist.

Run it with:
    uvicorn retailassist.api:app --reload
"""

import os
import secrets
import threading
import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .graph import ProviderMismatchError, RetailAssistant, sqlite_checkpointer
from .llm import LLM, LLMAuthError, LLMRateLimitError, LLMUnavailableError
from .llm import get_llm as llm_for_provider
from .rag import get_policy_index
from .store import ORDERS, PRODUCTS, find_order
from .tickets import InvalidOrderError, TicketCategory, TicketStatus, open_ticket, ticket_store

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

app = FastAPI(title="RetailAssist API", version="0.1.0")


# --- Dependencies ---------------------------------------------------------------------------------

@lru_cache
def get_llm() -> LLM:
    return llm_for_provider()


@lru_cache
def _checkpointer():
    return sqlite_checkpointer()


def get_assistant(llm: Annotated[LLM, Depends(get_llm)]) -> RetailAssistant:
    return RetailAssistant(llm, _checkpointer())


def require_staff_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Staff-only endpoints need the X-API-Key header to match RETAILASSIST_STAFF_API_KEY."""
    expected = os.environ.get("RETAILASSIST_STAFF_API_KEY")
    if not expected:
        raise HTTPException(503, "Staff endpoints are disabled: RETAILASSIST_STAFF_API_KEY is not set.")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Missing or invalid API key.")


# --- Chat sessions --------------------------------------------------------------------------------

# Conversations are saved by the graph's checkpointer. These locks stop two messages in the same
# conversation from running at once (striped, so memory doesn't grow with the number of sessions).
_session_locks = [threading.Lock() for _ in range(64)]


def _session_lock(session_id: str) -> threading.Lock:
    return _session_locks[hash(session_id) % len(_session_locks)]


# --- Models ---------------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Omit to start a new conversation; reuse the returned value to continue it.")


class Citation(BaseModel):
    number: int = Field(description="Matches the [n] markers in the reply")
    title: str | None
    source: str = Field(description="Policy file and section, e.g. data/policies/refund-policy.md#refunds")
    quotes: list[str] = Field(description="The exact policy passages the reply relied on")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    citations: list[Citation]


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
def health(llm: Annotated[LLM, Depends(get_llm)]) -> dict:
    return {"status": "ok", "provider": llm.name, "model": llm.model, "products": len(PRODUCTS), "orders": len(ORDERS),
            "policy_chunks": len(get_policy_index())}


@app.post("/chat")
def chat(request: ChatRequest, assistant: Annotated[RetailAssistant, Depends(get_assistant)]) -> ChatResponse:
    session_id = request.session_id
    if session_id is None:
        session_id = uuid.uuid4().hex
    elif not assistant.has_conversation(session_id):
        raise HTTPException(404, "Unknown session_id. Omit it to start a new conversation.")
    with _session_lock(session_id):  # one message at a time per conversation
        try:
            result = assistant.ask(session_id, request.message)
        except LLMAuthError as e:
            raise HTTPException(500, f"The server's model credentials aren't set up: {e}")
        except LLMRateLimitError:
            raise HTTPException(503, "The assistant is busy. Please try again in a moment.")
        except LLMUnavailableError:
            raise HTTPException(502, "The assistant couldn't be reached. Please try again.")
        except ProviderMismatchError as e:
            raise HTTPException(409, f"{e} Start a new conversation by omitting session_id.")
    return ChatResponse(session_id=session_id, **result)


@app.get("/orders/{order_id}")
def get_order(order_id: str, email: Annotated[str, Query(description="Email the order was placed with")]) -> dict:
    order = find_order(order_id, email)
    if order is None:
        raise HTTPException(404, "No order matches that order number and email.")
    return order


@app.post("/tickets", status_code=201)
def create_ticket(ticket: TicketCreate) -> Ticket:
    try:
        return Ticket(**open_ticket(email=ticket.email, category=ticket.category, message=ticket.message,
                                    order_id=ticket.order_id, source="api"))
    except InvalidOrderError as e:
        raise HTTPException(422, str(e))


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
