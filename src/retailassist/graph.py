"""The RetailAssist agent as a LangGraph graph.

Each customer message runs through:

    classify -> retrieve -> call_model -----------------------------------> respond
                                 ^      |
                                 |      v
                             tool_call <- approval_gate <- review_tool_calls

- classify:   the model labels the intent and pulls out order number, email, and product keywords.
- retrieve:   semantic search over the policy documents (vector database), plus matching products and
              the customer's order, attached to the customer's message as citable sources.
- call_model: the model answers, and may ask for tools.
- review_tool_calls: checks every requested tool call before anything runs. Order numbers must match
              the customer's email (access.py), and write actions (tickets, refunds) become requests
              for a person to approve (approvals.py).
- approval_gate: pauses the conversation with a LangGraph interrupt until staff approve or reject
              each request, then lets it continue (RetailAssistant.resume).
- tool_call:  runs the allowed tools, and approved write actions, and sends the results back to call_model.
- respond:    turns the model's last reply into the text for the customer, with numbered source citations.

The model provider (Gemini or Claude) is chosen with RETAILASSIST_LLM_PROVIDER; see llm/.

Conversation state is saved by a LangGraph checkpointer (SQLite by default), keyed by thread ID,
so a conversation picks up where it left off even after a restart.
"""

import json
import operator
import os
import re
import sqlite3
from dataclasses import asdict
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from .access import OTHER_CUSTOMER, OrderAccessError, authorize_order, normalize_email
from .approvals import ApprovalStore, approval_store, propose_refund, propose_ticket, public_view, run_action
from .llm import LLM, ToolCall, ToolResult, get_llm
from .rag import PolicyIndex, get_policy_index
from .store import DATA_DIR, PRODUCTS
from .tools import ALL_TOOLS, WRITE_TOOLS

MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are RetailAssist, a friendly shopping and support assistant for an outdoor gear store.

Help customers find products, check prices and stock, track their orders, and understand the store's \
return and shipping policies. Use your tools rather than guessing; if a tool finds nothing, say so. \
Answer questions about returns, refunds, and shipping only from the store's policy documents, and use \
the order's dates and status to say how the policy applies to that customer. If the policy documents \
don't cover a question, say so rather than filling the gap yourself. To look up an order, ask for the \
order number and the email it was placed with. Keep replies short and conversational.

Some customer messages come with policy sections and a <retrieved_context> block that the system looked \
up automatically for that message. Use them when they're relevant, and use your tools (including \
search_store_policies) for anything they don't cover.

You can't issue refunds, cancel orders, or change addresses yourself. When a customer wants a refund, \
tell them what the policy says and, if they confirm, use request_refund. For anything else a person has to \
do, offer a support ticket and use create_support_ticket once they agree. Both go to a team member to \
approve first; the conversation pauses until they decide, and the tool result tells you whether it was \
approved. Then tell the customer the outcome and give them any ticket ID.

Only discuss an order once a tool or the retrieved context has verified it against the customer's email. \
If a lookup is refused, pass the reason on and don't guess at the order's details."""

CLASSIFY_PROMPT = """You label customer messages for an outdoor gear store's support assistant.

Pick the intent of the newest message:
- product_question: finding products, prices, stock, sizing, recommendations
- order_status: where an order is, what's in it, delivery dates, tracking
- policy_question: general questions about returns, refunds, shipping options, or other store policies
- support_request: the customer wants something done: a return, refund, cancellation, address change, \
or a damaged or missing package
- other: greetings, thanks, or anything else

Also pull out an order number (like ORD-5001), an email address, and short product search keywords, \
if the newest message mentions them. Leave a field empty when the message doesn't mention it."""

Intent = Literal["product_question", "order_status", "policy_question", "support_request", "other"]


class Classification(BaseModel):
    intent: Intent
    order_id: str | None = Field(None, description="Order number, e.g. ORD-5001")
    email: str | None = Field(None, description="Customer email address")
    product_query: str | None = Field(None, description="A few keywords to search the catalog with")


class ConversationState(TypedDict, total=False):
    # Saved across turns. Messages are in the provider's own format and are only ever appended.
    provider: str
    messages: Annotated[list[dict], operator.add]
    customer_inputs: Annotated[list[str], operator.add]
    known_order_id: str | None
    known_email: str | None
    # The email this conversation has verified, and the orders verified with it (see access.py).
    verified_email: str | None
    verified_orders: list[str]
    # Set for each turn.
    user_input: str
    intent: Intent
    product_query: str | None
    new_order_details: bool
    policy_sources: list[str]
    context: str
    tool_rounds: int
    pending_tool_calls: list[dict]
    # From review_tool_calls, keyed by tool call ID: {"error": ...}, {"result": ...}, or {"approval": id}.
    call_reviews: dict[str, dict]
    stop_reason: str | None
    draft: str
    citations: list[dict] | None
    reply: str


class ProviderMismatchError(Exception):
    """A saved conversation was started with a different model provider than the one running now."""


class ApprovalPendingError(Exception):
    """The conversation is paused until a team member decides on a request."""

    def __init__(self, approvals: list[dict]):
        self.approvals = approvals
        super().__init__("This conversation is waiting for a team member to approve: "
                         + "; ".join(a["title"] for a in approvals))


def waiting_message(approvals: list[dict]) -> str:
    titles = "; ".join(a["title"] for a in approvals)
    return (f"Before I go ahead, a team member needs to approve this: {titles}. "
            "I'll pick up right here as soon as they've decided.")


# --- Retrieval helpers ----------------------------------------------------------------------------

STOPWORDS = {"the", "and", "for", "you", "your", "with", "can", "what", "how", "does", "are", "this", "that",
             "have", "from", "about", "will", "when", "they", "there", "would", "could", "order", "please"}


def _words(text: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) <= 2 or word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]  # crude plural folding: "refunds" matches "refund"
        words.add(word)
    return words


def search_catalog(query: str, limit: int = 5) -> list[dict]:
    """Return the products that share the most words with the query."""
    query_words = _words(query)
    scored = []
    for product in PRODUCTS.values():
        score = len(query_words & _words(f"{product['name']} {product['category']} {product['description']}"))
        if score:
            scored.append((score, product))
    scored.sort(key=lambda s: -s[0])
    return [product for _, product in scored[:limit]]


# --- Citations from [section-id] markers ----------------------------------------------------------

_SECTION_ID = r"[a-z0-9-]+#[a-z0-9-]+"
_MARKER = re.compile(rf"\s?\[({_SECTION_ID}(?:\s*,\s*{_SECTION_ID})*)\]")


def citations_from_markers(text: str, policy_index: PolicyIndex) -> tuple[str, list[dict]]:
    """Replace [section-id] markers with numbered [n] markers, keeping only IDs that exist in the index.

    Used for providers without native citations. Unknown IDs are dropped rather than shown.
    """
    sources: dict[str, dict] = {}

    def replace(match: re.Match) -> str:
        numbers = []
        for section_id in re.split(r"\s*,\s*", match.group(1)):
            chunk = policy_index.chunk(section_id)
            if chunk is None:
                continue
            source = sources.setdefault(chunk.id, {"number": len(sources) + 1, "title": chunk.title,
                                                   "source": chunk.source, "quotes": []})
            if source["number"] not in numbers:
                numbers.append(source["number"])
        return "".join(f"[{n}]" for n in numbers)

    return _MARKER.sub(replace, text).strip(), list(sources.values())


# --- Graph ----------------------------------------------------------------------------------------

def _clients(config: RunnableConfig) -> tuple[str, ...]:
    """Keys that failed order lookups are rate-limited under: the conversation, and the client's IP if known."""
    configurable = config.get("configurable", {})
    keys = [f"session:{configurable['thread_id']}"]
    if configurable.get("client"):
        keys.append(f"ip:{configurable['client']}")
    return tuple(keys)


class _Access:
    """Order checks within one node, carrying newly verified details forward to the state update."""

    def __init__(self, state: ConversationState, config: RunnableConfig):
        self.clients = _clients(config)
        self.email = state.get("verified_email")
        self.orders = list(state.get("verified_orders") or [])

    def order(self, order_id: str, email: str) -> dict:
        order = authorize_order(order_id, email, clients=self.clients, verified_email=self.email)
        self.email = normalize_email(email)
        if order["order_id"] not in self.orders:
            self.orders.append(order["order_id"])
        return order

    def check_email(self, email: str) -> None:
        if self.email and normalize_email(email) != self.email:
            raise OrderAccessError(OTHER_CUSTOMER)

    def update(self) -> dict:
        return {"verified_email": self.email, "verified_orders": self.orders}


def build_graph(llm: LLM, checkpointer: BaseCheckpointSaver | None = None, policy_index: PolicyIndex | None = None,
                approvals: ApprovalStore | None = None):
    policy_index = policy_index or get_policy_index()
    approvals = approvals or approval_store
    tools_by_name = {tool.name: tool for tool in ALL_TOOLS}
    system_prompt = SYSTEM_PROMPT + llm.instructions

    def start_turn(state: ConversationState) -> dict:
        provider = state.get("provider")
        if provider and provider != llm.name:
            raise ProviderMismatchError(
                f"This conversation was started with {provider}, but the server is running {llm.name}.")
        return {"provider": llm.name, "customer_inputs": [state["user_input"]], "tool_rounds": 0,
                "pending_tool_calls": [], "stop_reason": None, "draft": "", "citations": [], "policy_sources": []}

    def classify(state: ConversationState) -> dict:
        recent = state.get("customer_inputs", [])[-3:]
        known = {"order_id": state.get("known_order_id"), "email": state.get("known_email")}
        prompt = (f"Earlier customer messages: {json.dumps(recent[:-1])}\n"
                  f"Details already known: {json.dumps(known)}\n\n"
                  f"Newest message: {state['user_input']}")
        result = llm.classify(CLASSIFY_PROMPT, prompt, Classification) or Classification(intent="other")
        update: dict = {"intent": result.intent, "product_query": result.product_query,
                        "new_order_details": bool(result.order_id or result.email)}
        # Remember order numbers and emails for later turns.
        if result.order_id:
            update["known_order_id"] = result.order_id.strip().upper()
        if result.email:
            update["known_email"] = result.email.strip().lower()
        return update

    def retrieve(state: ConversationState, config: RunnableConfig) -> dict:
        intent = state["intent"]
        user_input = state["user_input"]
        parts = []
        policy_hits = []
        if intent in {"policy_question", "support_request", "order_status"}:
            policy_hits = policy_index.search(user_input)

        if intent == "product_question":
            products = search_catalog(state.get("product_query") or user_input)
            if products:
                parts.append("Possibly relevant products:\n" + json.dumps(products, ensure_ascii=False))

        access = _Access(state, config)
        order_id, email = state.get("known_order_id"), state.get("known_email")
        # Look the order up when the customer has just given its details, or it's already verified.
        # (Not on every turn, so one wrong pair isn't counted as a failed attempt again and again.)
        if (intent in {"order_status", "support_request"} and order_id and email
                and (state.get("new_order_details") or order_id in access.orders)):
            try:
                order = access.order(order_id, email)
                parts.append("The customer's order (verified against their email):\n" + json.dumps(order))
            except OrderAccessError as e:
                parts.append(f"Order lookup for {order_id} refused: {e}")

        context = "\n\n".join(parts)
        return {"context": context, "policy_sources": [hit.chunk.source for hit in policy_hits],
                "messages": [llm.user_message(user_input, context, policy_hits)], **access.update()}

    def call_model(state: ConversationState) -> dict:
        turn = llm.respond(system_prompt, state["messages"], ALL_TOOLS)
        update: dict = {"stop_reason": turn.stop_reason, "draft": turn.text, "citations": turn.citations,
                        "pending_tool_calls": [asdict(call) for call in turn.tool_calls]}
        if turn.message is not None:
            update["messages"] = [turn.message]
        return update

    def route_after_model(state: ConversationState) -> Literal["review_tool_calls", "respond"]:
        if state["stop_reason"] == "tool_use" and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
            return "review_tool_calls"
        return "respond"

    def review_tool_calls(state: ConversationState, config: RunnableConfig) -> dict:
        """Check order access for every call, and turn write actions into requests for approval."""
        access = _Access(state, config)
        reviews: dict[str, dict] = {}
        for call in (ToolCall(**c) for c in state["pending_tool_calls"]):
            try:
                reviews[call.id] = review(call, access, config)
            except OrderAccessError as e:
                reviews[call.id] = {"error": str(e)}
            except ValidationError as e:
                reviews[call.id] = {"error": f"Invalid arguments for {call.name}: {e.errors(include_url=False)}"}
        return {"call_reviews": reviews, **access.update()}

    def review(call: ToolCall, access: _Access, config: RunnableConfig) -> dict:
        args = call.args
        if call.name not in WRITE_TOOLS:
            if args.get("order_id"):
                access.order(args["order_id"], args.get("email", ""))
            return {}
        action = WRITE_TOOLS[call.name](args)
        order = access.order(action.order_id, action.email) if action.order_id else None
        access.check_email(action.email)
        if call.name == "request_refund":
            outcome, proposal = propose_refund(action, order)
            if proposal is None:  # not eligible, or already requested: nothing to write, so nothing to approve
                return {"result": json.dumps(outcome)}
        else:
            proposal = propose_ticket(action, order)
        approval = approvals.submit(proposal, source="chat", thread_id=config["configurable"]["thread_id"],
                                    call_id=call.id)
        return {"approval": approval["id"]}

    def route_after_review(state: ConversationState) -> Literal["approval_gate", "tool_call"]:
        if any("approval" in r for r in state["call_reviews"].values()):
            return "approval_gate"
        return "tool_call"

    def approval_gate(state: ConversationState) -> dict:
        """Pause until every requested write action is approved or rejected.

        Decisions are saved in the approval store; the API resumes the graph once they're all made. This
        node re-runs from the top on resume, so it must not change anything before interrupt().
        """
        ids = [r["approval"] for r in state["call_reviews"].values() if "approval" in r]
        while pending := [a for a in map(approvals.get, ids) if a["status"] == "pending"]:
            interrupt({"approvals": [public_view(a) for a in pending]})
        return {}

    def tool_call(state: ConversationState) -> dict:
        results = []
        reviews = state.get("call_reviews") or {}
        for call in (ToolCall(**c) for c in state["pending_tool_calls"]):
            review = reviews.get(call.id, {})
            tool = tools_by_name.get(call.name)
            try:
                if "error" in review:
                    results.append(ToolResult(call, review["error"], is_error=True))
                elif "result" in review:
                    results.append(ToolResult(call, review["result"]))
                elif "approval" in review:
                    results.append(ToolResult(call, json.dumps(decided(approvals.get(review["approval"])))))
                elif tool is None or call.name in WRITE_TOOLS:
                    raise ValueError(f"Unknown tool: {call.name}")
                else:
                    results.append(ToolResult(call, tool.call(call.args)))
            except Exception as e:  # report tool failures to the model instead of crashing the turn
                results.append(ToolResult(call, str(e), is_error=True))
        return {"messages": [llm.tool_results_message(results)], "pending_tool_calls": [], "call_reviews": {},
                "tool_rounds": state.get("tool_rounds", 0) + 1}

    def decided(approval: dict) -> dict:
        """Run an approved action, and say what happened either way."""
        view = {"request_id": approval["approval_id"], "approved": approval["status"] == "approved",
                "note_from_team": approval["note"]}
        if approval["status"] != "approved":
            return view | {"message": "A team member declined this request, so nothing was changed."}
        # On a retried run the action has already happened; don't do it twice.
        return view | (approval["result"] or run_action(approval, approvals))

    def respond(state: ConversationState) -> dict:
        update: dict = {"pending_tool_calls": []}
        # If we stopped with tool calls still pending (tool-round limit or max_tokens), close them out
        # so the next turn's history is valid.
        pending = [ToolCall(**c) for c in state.get("pending_tool_calls", [])]
        if pending and state["stop_reason"] != "refusal":
            update["messages"] = [llm.tool_results_message(
                [ToolResult(call, "Not run: tool limit reached.", is_error=True) for call in pending])]

        if state["stop_reason"] == "refusal":
            update.update(reply="Sorry, I can't help with that request.", citations=[])
        elif state.get("draft") and state["stop_reason"] != "error":
            if state.get("citations") is None:  # provider has no native citations
                update["reply"], update["citations"] = citations_from_markers(state["draft"], policy_index)
            else:
                update["reply"] = state["draft"]
        else:
            update.update(reply="Sorry, I wasn't able to finish that. Could you try asking another way?", citations=[])
        return update

    builder = StateGraph(ConversationState)
    builder.add_node("start_turn", start_turn)
    builder.add_node("classify", classify)
    builder.add_node("retrieve", retrieve)
    builder.add_node("call_model", call_model)
    builder.add_node("review_tool_calls", review_tool_calls)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("tool_call", tool_call)
    builder.add_node("respond", respond)
    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "call_model")
    builder.add_conditional_edges("call_model", route_after_model, ["review_tool_calls", "respond"])
    builder.add_conditional_edges("review_tool_calls", route_after_review, ["approval_gate", "tool_call"])
    builder.add_edge("approval_gate", "tool_call")
    builder.add_edge("tool_call", "call_model")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


# --- Public interface -----------------------------------------------------------------------------

def sqlite_checkpointer(path: str | os.PathLike | None = None) -> SqliteSaver:
    path = path or os.environ.get("RETAILASSIST_CONVERSATIONS_DB", DATA_DIR / "conversations.db")
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class RetailAssistant:
    """Runs customer messages through the graph, saving each conversation under its thread ID."""

    def __init__(self, llm: LLM | None = None, checkpointer: BaseCheckpointSaver | None = None,
                 policy_index: PolicyIndex | None = None, approvals: ApprovalStore | None = None):
        self.llm = llm or get_llm()
        self.approvals = approvals or approval_store
        self.graph = build_graph(self.llm, checkpointer or sqlite_checkpointer(), policy_index, self.approvals)

    def ask(self, thread_id: str, user_text: str, client: str | None = None) -> dict:
        """Send one customer message. Returns the reply, the intent it was classified as, the policy
        sources the reply cites (numbered to match the [n] markers in the reply), and any requests
        waiting for a team member's approval.

        client: the customer's IP address or similar, for rate-limiting failed order lookups.
        Raises ApprovalPendingError while the conversation is paused for approval.
        """
        if self._paused(thread_id):
            if waiting := self.pending_approvals(thread_id):
                raise ApprovalPendingError(waiting)
            self.resume(thread_id)  # decided, but the earlier resume didn't finish (e.g. the model was down)
        self.graph.invoke({"user_input": user_text}, self._config(thread_id, client))
        return self.result(thread_id)

    def resume(self, thread_id: str) -> dict:
        """Continue a paused conversation once its requests are decided. Returns the new reply."""
        if self._paused(thread_id):
            snapshot = self.graph.get_state(self._config(thread_id))
            # After an interrupt, resume it; after an error partway through, pick up from the last step.
            self.graph.invoke(Command(resume=True) if snapshot.interrupts else None, self._config(thread_id))
        return self.result(thread_id)

    def result(self, thread_id: str) -> dict:
        """The conversation's latest reply, or a waiting message if it's paused for approval."""
        state = self.graph.get_state(self._config(thread_id)).values
        waiting = self.pending_approvals(thread_id)
        if waiting:
            return {"reply": waiting_message(waiting), "intent": state.get("intent", "other"), "citations": [],
                    "pending_approvals": [public_view(a) for a in waiting]}
        return {"reply": state.get("reply", ""), "intent": state.get("intent", "other"),
                "citations": state.get("citations") or [], "pending_approvals": []}

    def pending_approvals(self, thread_id: str) -> list[dict]:
        if not self.graph.get_state(self._config(thread_id)).interrupts:
            return []
        return self.approvals.list(status="pending", thread_id=thread_id)

    def has_conversation(self, thread_id: str) -> bool:
        return bool(self.graph.get_state(self._config(thread_id)).values)

    def _paused(self, thread_id: str) -> bool:
        return bool(self.graph.get_state(self._config(thread_id)).next)

    @staticmethod
    def _config(thread_id: str, client: str | None = None) -> dict:
        return {"configurable": {"thread_id": thread_id, "client": client}}
