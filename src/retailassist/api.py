"""HTTP API for RetailAssist.

Run it with:
    uvicorn retailassist.api:app --reload
"""

import os
import secrets
import threading
import uuid
from functools import lru_cache
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .access import OrderAccessError, TooManyAttemptsError, authorize_order
from .approvals import ApprovalStatus, approval_store, run_action
from .graph import ApprovalPendingError, ProviderMismatchError, RetailAssistant, sqlite_checkpointer
from .llm import LLM, LLMAuthError, LLMRateLimitError, LLMUnavailableError
from .llm import get_llm as llm_for_provider
from .observability import get_observability
from .rag import get_policy_index
from .store import ORDERS, PRODUCTS
from .tickets import InvalidOrderError, TicketCategory, TicketStatus, open_ticket, ticket_store

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_observability().shutdown()  # send any traces still buffered


app = FastAPI(title="RetailAssist API", version="0.1.0", lifespan=lifespan)


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


def _client(request: Request) -> str | None:
    """The caller's address, for rate-limiting failed order lookups. Behind a proxy or load balancer,
    run uvicorn with --proxy-headers so this is the customer's address rather than the proxy's."""
    return request.client.host if request.client else None


@contextmanager
def _model_errors():
    """Turn model and conversation errors into HTTP errors."""
    try:
        yield
    except LLMAuthError as e:
        raise HTTPException(500, f"The server's model credentials aren't set up: {e}")
    except LLMRateLimitError:
        raise HTTPException(503, "The assistant is busy. Please try again in a moment.")
    except LLMUnavailableError:
        raise HTTPException(502, "The assistant couldn't be reached. Please try again.")
    except ProviderMismatchError as e:
        raise HTTPException(409, f"{e} Start a new conversation by omitting session_id.")
    except ApprovalPendingError as e:
        raise HTTPException(409, f"{e}. Check back with GET /chat/{{session_id}}.")


# --- Models ---------------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Omit to start a new conversation; reuse the returned value to continue it.")


class Citation(BaseModel):
    number: int = Field(description="Matches the [n] markers in the reply")
    title: str | None
    source: str = Field(description="Policy file and section, e.g. data/policies/refund-policy.md#refunds")
    quotes: list[str] = Field(description="The exact policy passages the reply relied on")


class PendingRequest(BaseModel):
    request_id: str
    title: str
    status: ApprovalStatus
    note: str
    result: dict | None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    citations: list[Citation]
    pending_approvals: list[PendingRequest] = Field(
        description="Requests waiting for a team member. While any are listed, the conversation is paused; "
                    "poll GET /chat/{session_id} for the reply once they're decided.")
    trace_id: str | None = Field(None, description="Langfuse trace for this reply, when tracing is on")


class Approval(BaseModel):
    approval_id: str
    id: int
    created_at: str
    decided_at: str | None
    source: str = Field(description="chat or mcp")
    thread_id: str | None = Field(description="The chat session waiting on this, if any")
    action: str
    args: dict
    title: str
    details: str
    customer_email: str
    order_id: str | None
    status: ApprovalStatus
    note: str
    result: dict | None = Field(description="What the action did, once approved and run")


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field("", max_length=2000, description="Passed on to the customer")


class DecisionResult(BaseModel):
    approval: Approval
    customer_reply: str | None = Field(description="The assistant's reply to the customer, for chat requests")


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
    return {"status": "ok", "provider": llm.name, "model": llm.model, "tracing": get_observability().enabled,
            "products": len(PRODUCTS), "orders": len(ORDERS),
            "policy_chunks": len(get_policy_index())}


@app.post("/chat")
def chat(request: ChatRequest, http: Request,
         assistant: Annotated[RetailAssistant, Depends(get_assistant)]) -> ChatResponse:
    session_id = request.session_id
    if session_id is None:
        session_id = uuid.uuid4().hex
    elif not assistant.has_conversation(session_id):
        raise HTTPException(404, "Unknown session_id. Omit it to start a new conversation.")
    with _session_lock(session_id), _model_errors():  # one message at a time per conversation
        result = assistant.ask(session_id, request.message, client=_client(http))
    return ChatResponse(session_id=session_id, **result)


@app.get("/chat/{session_id}")
def chat_status(session_id: str, assistant: Annotated[RetailAssistant, Depends(get_assistant)]) -> ChatResponse:
    """The conversation's latest reply. While it's waiting for approval, lists what it's waiting on."""
    if not assistant.has_conversation(session_id):
        raise HTTPException(404, "Unknown session_id.")
    return ChatResponse(session_id=session_id, **assistant.result(session_id))


def _authorize(order_id: str, email: str, request: Request) -> dict:
    client = _client(request)
    try:
        return authorize_order(order_id, email, clients=(f"ip:{client}",) if client else ())
    except TooManyAttemptsError as e:
        raise HTTPException(429, str(e))


@app.get("/orders/{order_id}")
def get_order(order_id: str, request: Request,
              email: Annotated[str, Query(description="Email the order was placed with")]) -> dict:
    try:
        return _authorize(order_id, email, request)
    except OrderAccessError:
        raise HTTPException(404, "No order matches that order number and email.")


@app.post("/tickets", status_code=201)
def create_ticket(ticket: TicketCreate, request: Request) -> Ticket:
    """For customers filling in a support form themselves. Tickets the assistant opens go through /approvals."""
    try:
        if ticket.order_id:
            _authorize(ticket.order_id, ticket.email, request)
        return Ticket(**open_ticket(email=ticket.email, category=ticket.category, message=ticket.message,
                                    order_id=ticket.order_id, source="api"))
    except (OrderAccessError, InvalidOrderError) as e:
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


# --- Approvals (staff) ----------------------------------------------------------------------------

@app.get("/approvals", dependencies=[Depends(require_staff_key)])
def list_approvals(status: ApprovalStatus | None = "pending") -> list[Approval]:
    """Write actions waiting for (or past) a decision. Pending ones by default."""
    return [Approval(**a) for a in approval_store.list(status)]


@app.get("/approvals/{approval_id}", dependencies=[Depends(require_staff_key)])
def get_approval(approval_id: int) -> Approval:
    approval = approval_store.get(approval_id)
    if approval is None:
        raise HTTPException(404, "Approval not found.")
    return Approval(**approval)


@app.post("/approvals/{approval_id}", dependencies=[Depends(require_staff_key)])
def decide_approval(approval_id: int, decision: Decision,
                    assistant: Annotated[RetailAssistant, Depends(get_assistant)]) -> DecisionResult:
    """Approve or reject a request. Approved actions run straight away; a waiting chat picks up again.

    Sending the same decision again is safe, and retries a chat that couldn't continue the first time.
    """
    status = "approved" if decision.decision == "approve" else "rejected"
    approval = approval_store.get(approval_id)
    if approval is None:
        raise HTTPException(404, "Approval not found.")
    if approval["status"] == "pending":
        approval = approval_store.decide(approval_id, status, decision.note) or approval_store.get(approval_id)
    if approval["status"] != status:
        raise HTTPException(409, f"{approval['approval_id']} was already {approval['status']}.")

    if approval["thread_id"] is None:  # from the MCP server: run it here
        if status == "approved" and approval["result"] is None:
            try:
                run_action(approval)
            except PermissionError as e:
                raise HTTPException(409, f"Approved, but it can't run: {e}")
        return DecisionResult(approval=Approval(**approval_store.get(approval_id)), customer_reply=None)

    # From a chat: continue the conversation once nothing in it is still waiting.
    thread_id = approval["thread_id"]
    reply = None
    with _session_lock(thread_id), _model_errors():
        if not approval_store.list(status="pending", thread_id=thread_id):
            reply = assistant.resume(thread_id)["reply"]
    return DecisionResult(approval=Approval(**approval_store.get(approval_id)), customer_reply=reply)
