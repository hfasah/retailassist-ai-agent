"""The RetailAssist agent as a LangGraph graph.

Each customer message runs through:

    classify -> retrieve -> call_model -> respond
                                 |  ^
                                 v  |
                              tool_call

- classify:   Claude labels the intent and pulls out order number, email, and product keywords.
- retrieve:   semantic search over the policy documents (vector database), plus matching products and
              the customer's order. Policy sections go in as search_result blocks so Claude can cite them.
- call_model: Claude answers, and may ask for tools.
- tool_call:  runs the requested tools and sends the results back to call_model.
- respond:    turns Claude's last reply into the text for the customer, with numbered source citations.

Conversation state is saved by a LangGraph checkpointer (SQLite by default), keyed by thread ID,
so a conversation picks up where it left off even after a restart.
"""

import json
import operator
import os
import re
import sqlite3
from typing import Annotated, Literal, TypedDict

import anthropic
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .rag import PolicyIndex, get_policy_index
from .store import DATA_DIR, PRODUCTS
from .tools import ALL_TOOLS, find_order

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 8
# If a safety classifier declines, retry on Anthropic's recommended fallback model.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

SYSTEM_PROMPT = """You are RetailAssist, a friendly shopping and support assistant for an outdoor gear store.

Help customers find products, check prices and stock, track their orders, and understand the store's \
return and shipping policies. Use your tools rather than guessing; if a tool finds nothing, say so. \
Answer questions about returns, refunds, and shipping only from the store's policy documents, and use \
the order's dates and status to say how the policy applies to that customer. If the policy documents \
don't cover a question, say so rather than filling the gap yourself. To look up an order, ask for the \
order number and the email it was placed with. Keep replies short and conversational.

Some customer messages come with policy search results and a <retrieved_context> block that the system \
looked up automatically for that message. Use them when they're relevant, and use your tools (including \
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
    # Saved across turns. Messages are only ever appended, never edited.
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
    stop_reason: str | None
    draft: str
    citations: list[dict]
    reply: str


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


# --- Graph ----------------------------------------------------------------------------------------

def build_graph(client: anthropic.Anthropic, checkpointer: BaseCheckpointSaver | None = None,
                policy_index: PolicyIndex | None = None):
    policy_index = policy_index or get_policy_index()
    tools_by_name = {tool.name: tool for tool in ALL_TOOLS}
    tool_definitions = [tool.to_dict() for tool in ALL_TOOLS]

    def classify(state: ConversationState) -> dict:
        recent = state.get("customer_inputs", [])[-3:]
        known = {"order_id": state.get("known_order_id"), "email": state.get("known_email")}
        prompt = (f"Earlier customer messages: {json.dumps(recent[:-1])}\n"
                  f"Details already known: {json.dumps(known)}\n\n"
                  f"Newest message: {state['user_input']}")
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=CLASSIFY_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=Classification,
            output_config={"effort": "low"},
            **FALLBACK,
        )
        result = response.parsed_output or Classification(intent="other")
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
        content: list[dict] = [hit.to_search_result() for hit in policy_hits]
        if context:
            content.append({"type": "text", "text": f"<retrieved_context>\n{context}\n</retrieved_context>"})
        content.append({"type": "text", "text": user_input})
        return {"context": context, "policy_sources": [hit.chunk.source for hit in policy_hits],
                "messages": [{"role": "user", "content": content}]}

    def call_model(state: ConversationState) -> dict:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tool_definitions,
            messages=state["messages"],
            output_config={"effort": "medium"},
            **FALLBACK,
        )
        draft, citations = _text_with_citations(response.content)
        update: dict = {"stop_reason": response.stop_reason, "draft": draft, "citations": citations}
        # A refused or empty reply can't go back into the history; the API rejects empty assistant turns.
        if response.stop_reason != "refusal" and response.content:
            # Plain dicts, exactly as the API returned them, so the checkpointer can save them.
            content = [block.model_dump(mode="json", exclude_unset=True) for block in response.content]
            update["messages"] = [{"role": "assistant", "content": content}]
        return update

    def route_after_model(state: ConversationState) -> Literal["tool_call", "respond"]:
        if state["stop_reason"] == "tool_use" and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
            return "tool_call"
        return "respond"

    def tool_call(state: ConversationState) -> dict:
        results = []
        for block in _unanswered_tool_uses(state["messages"]):
            tool = tools_by_name.get(block["name"])
            try:
                if tool is None:
                    raise ValueError(f"Unknown tool: {block['name']}")
                output = tool.call(block["input"])
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": output})
            except Exception as e:  # report tool failures to Claude instead of crashing the turn
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": str(e), "is_error": True})
        return {"messages": [{"role": "user", "content": results}], "tool_rounds": state.get("tool_rounds", 0) + 1}

    def respond(state: ConversationState) -> dict:
        update: dict = {}
        # If we stopped with tool calls still pending (tool-round limit or max_tokens), close them out
        # so the next turn's history is valid.
        pending = _unanswered_tool_uses(state["messages"])
        if pending:
            update["messages"] = [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b["id"], "content": "Not run: tool limit reached.", "is_error": True}
                for b in pending]}]

        if state["stop_reason"] == "refusal":
            update.update(reply="Sorry, I can't help with that request.", citations=[])
        elif state.get("draft"):
            update["reply"] = state["draft"]
        else:
            update.update(reply="Sorry, I wasn't able to finish that. Could you try asking another way?", citations=[])
        return update

    def start_turn(state: ConversationState) -> dict:
        return {"customer_inputs": [state["user_input"]], "tool_rounds": 0, "stop_reason": None, "draft": "",
                "citations": [], "policy_sources": []}

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


def _text_with_citations(content) -> tuple[str, list[dict]]:
    """Join Claude's text blocks, adding [1]-style markers where it cited a policy passage.

    Returns the text and the list of sources, numbered in the order they're first cited.
    """
    sources: dict[str, dict] = {}
    text = ""
    for block in content:
        if block.type != "text":
            continue
        text += block.text
        markers = []
        for citation in block.citations or []:
            if citation.type != "search_result_location":
                continue
            source = sources.setdefault(citation.source, {
                "number": len(sources) + 1, "title": citation.title, "source": citation.source, "quotes": []})
            if citation.cited_text not in source["quotes"]:
                source["quotes"].append(citation.cited_text)
            if source["number"] not in markers:
                markers.append(source["number"])
        if markers:
            # Put markers before any trailing whitespace, so they sit right after the cited sentence.
            stripped = text.rstrip()
            text = stripped + "".join(f"[{n}]" for n in markers) + text[len(stripped):]
    return text.strip(), list(sources.values())


def _unanswered_tool_uses(messages: list[dict]) -> list[dict]:
    """Tool calls in the last message, if it's an assistant message (so no results have been sent yet)."""
    if not messages or messages[-1]["role"] != "assistant":
        return []
    content = messages[-1]["content"]
    if isinstance(content, str):
        return []
    return [block for block in content if block.get("type") == "tool_use"]


# --- Public interface -----------------------------------------------------------------------------

def sqlite_checkpointer(path: str | os.PathLike | None = None) -> SqliteSaver:
    path = path or os.environ.get("RETAILASSIST_CONVERSATIONS_DB", DATA_DIR / "conversations.db")
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class RetailAssistant:
    """Runs customer messages through the graph, saving each conversation under its thread ID."""

    def __init__(self, client: anthropic.Anthropic | None = None, checkpointer: BaseCheckpointSaver | None = None,
                 policy_index: PolicyIndex | None = None):
        self.graph = build_graph(client or anthropic.Anthropic(), checkpointer or sqlite_checkpointer(), policy_index)

    def ask(self, thread_id: str, user_text: str) -> dict:
        """Send one customer message. Returns the reply, the intent it was classified as, and the policy
        sources the reply cites (numbered to match the [n] markers in the reply)."""
        state = self.graph.invoke({"user_input": user_text}, {"configurable": {"thread_id": thread_id}})
        return {"reply": state["reply"], "intent": state["intent"], "citations": state.get("citations", [])}

    def has_conversation(self, thread_id: str) -> bool:
        return bool(self.graph.get_state({"configurable": {"thread_id": thread_id}}).values)
