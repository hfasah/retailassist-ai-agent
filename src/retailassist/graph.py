"""The RetailAssist agent as a LangGraph graph.

Each customer message runs through:

    classify -> retrieve -> call_model -> respond
                                 |  ^
                                 v  |
                              tool_call

- classify:   the model labels the intent and pulls out order number, email, and product keywords.
- retrieve:   semantic search over the policy documents (vector database), plus matching products and
              the customer's order, attached to the customer's message as citable sources.
- call_model: the model answers, and may ask for tools.
- tool_call:  runs the requested tools and sends the results back to call_model.
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
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .llm import LLM, ToolCall, ToolResult, get_llm
from .rag import PolicyIndex, get_policy_index
from .store import DATA_DIR, PRODUCTS
from .tools import ALL_TOOLS, find_order

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

You can't start returns, issue refunds, cancel orders, or change addresses yourself. When a customer \
wants one of those, tell them whether the policy allows it, and offer to open a support ticket so a team \
member can follow up. Only open a ticket after the customer agrees, and give them the ticket ID."""

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
    # Set for each turn.
    user_input: str
    intent: Intent
    product_query: str | None
    policy_sources: list[str]
    context: str
    tool_rounds: int
    pending_tool_calls: list[dict]
    stop_reason: str | None
    draft: str
    citations: list[dict] | None
    reply: str


class ProviderMismatchError(Exception):
    """A saved conversation was started with a different model provider than the one running now."""


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

def build_graph(llm: LLM, checkpointer: BaseCheckpointSaver | None = None, policy_index: PolicyIndex | None = None):
    policy_index = policy_index or get_policy_index()
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
        update: dict = {"intent": result.intent, "product_query": result.product_query}
        # Remember order numbers and emails for later turns.
        if result.order_id:
            update["known_order_id"] = result.order_id.strip().upper()
        if result.email:
            update["known_email"] = result.email.strip().lower()
        return update

    def retrieve(state: ConversationState) -> dict:
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

        if intent in {"order_status", "support_request"} and state.get("known_order_id") and state.get("known_email"):
            order = find_order(state["known_order_id"], state["known_email"])
            if order:
                parts.append("The customer's order (verified against their email):\n" + json.dumps(order))
            else:
                parts.append(f"No order matches {state['known_order_id']} and {state['known_email']}.")

        context = "\n\n".join(parts)
        return {"context": context, "policy_sources": [hit.chunk.source for hit in policy_hits],
                "messages": [llm.user_message(user_input, context, policy_hits)]}

    def call_model(state: ConversationState) -> dict:
        turn = llm.respond(system_prompt, state["messages"], ALL_TOOLS)
        update: dict = {"stop_reason": turn.stop_reason, "draft": turn.text, "citations": turn.citations,
                        "pending_tool_calls": [asdict(call) for call in turn.tool_calls]}
        if turn.message is not None:
            update["messages"] = [turn.message]
        return update

    def route_after_model(state: ConversationState) -> Literal["tool_call", "respond"]:
        if state["stop_reason"] == "tool_use" and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
            return "tool_call"
        return "respond"

    def tool_call(state: ConversationState) -> dict:
        results = []
        for call in (ToolCall(**c) for c in state["pending_tool_calls"]):
            tool = tools_by_name.get(call.name)
            try:
                if tool is None:
                    raise ValueError(f"Unknown tool: {call.name}")
                results.append(ToolResult(call, tool.call(call.args)))
            except Exception as e:  # report tool failures to the model instead of crashing the turn
                results.append(ToolResult(call, str(e), is_error=True))
        return {"messages": [llm.tool_results_message(results)], "pending_tool_calls": [],
                "tool_rounds": state.get("tool_rounds", 0) + 1}

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
    builder.add_node("tool_call", tool_call)
    builder.add_node("respond", respond)
    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "call_model")
    builder.add_conditional_edges("call_model", route_after_model, ["tool_call", "respond"])
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
                 policy_index: PolicyIndex | None = None):
        self.llm = llm or get_llm()
        self.graph = build_graph(self.llm, checkpointer or sqlite_checkpointer(), policy_index)

    def ask(self, thread_id: str, user_text: str) -> dict:
        """Send one customer message. Returns the reply, the intent it was classified as, and the policy
        sources the reply cites (numbered to match the [n] markers in the reply)."""
        state = self.graph.invoke({"user_input": user_text}, {"configurable": {"thread_id": thread_id}})
        return {"reply": state["reply"], "intent": state["intent"], "citations": state.get("citations") or []}

    def has_conversation(self, thread_id: str) -> bool:
        return bool(self.graph.get_state({"configurable": {"thread_id": thread_id}}).values)
