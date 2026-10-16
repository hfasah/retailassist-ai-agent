import json
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from retailassist.graph import MAX_TOOL_ROUNDS, RetailAssistant, search_catalog, search_policies, sqlite_checkpointer
from retailassist.store import ORDERS

EMAIL = ORDERS["ORD-5003"]["customer"]["email"]


def new_thread() -> str:
    return uuid.uuid4().hex


def test_search_policies_finds_relevant_sections():
    results = search_policies("how long until my refund shows up on my card")
    assert any("Refunds" in r for r in results)


def test_search_catalog_ranks_matches():
    results = search_catalog("waterproof hiking boots")
    assert results[0]["sku"] == "SKU-1001"


def test_order_turn_runs_every_node(fake_claude):
    fake_claude.classification = {"intent": "order_status", "order_id": "ord-5003", "email": EMAIL, "product_query": None}
    fake_claude.replies = [{"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}},
                           "It's on its way!"]
    assistant = RetailAssistant(fake_claude.client(), InMemorySaver())

    result = assistant.ask(new_thread(), "Where is ORD-5003? I'm " + EMAIL)

    assert result == {"reply": "It's on its way!", "intent": "order_status"}
    # retrieve: the order and policy sections were added next to the customer's message.
    first_user_message = fake_claude.requests[0]["messages"][0]["content"]
    assert "<retrieved_context>" in first_user_message[0]["text"]
    assert '"order_id": "ORD-5003"' in first_user_message[0]["text"]
    assert first_user_message[-1]["text"].startswith("Where is ORD-5003?")
    # tool_call: the tool ran and its result went back to Claude.
    tool_result = fake_claude.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert json.loads(tool_result["content"])["order_id"] == "ORD-5003"


def test_product_question_retrieves_products(fake_claude):
    fake_claude.classification = {"intent": "product_question", "order_id": None, "email": None, "product_query": "rain jacket"}
    assistant = RetailAssistant(fake_claude.client(), InMemorySaver())
    assistant.ask(new_thread(), "Got anything for rainy hikes?")
    context = fake_claude.requests[0]["messages"][0]["content"][0]["text"]
    assert "Stormshield Rain Jacket" in context


def test_conversation_persists_across_restarts(fake_claude, tmp_path):
    db = tmp_path / "conversations.db"
    thread = new_thread()

    fake_claude.classification = {"intent": "order_status", "order_id": "ORD-5003", "email": EMAIL, "product_query": None}
    first = RetailAssistant(fake_claude.client(), sqlite_checkpointer(db))
    first.ask(thread, "Where's ORD-5003? Email " + EMAIL)

    # A new assistant with a new database connection, like after a server restart.
    fake_claude.classification = {"intent": "support_request", "order_id": None, "email": None, "product_query": None}
    second = RetailAssistant(fake_claude.client(), sqlite_checkpointer(db))
    assert second.has_conversation(thread)
    assert not second.has_conversation(new_thread())
    second.ask(thread, "I want to return it")

    messages = fake_claude.requests[-1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    # The order number and email from turn one were remembered, so turn two still retrieved the order.
    assert '"order_id": "ORD-5003"' in messages[-1]["content"][0]["text"]
    # The classifier saw the earlier message too.
    assert "Where's ORD-5003?" in fake_claude.classify_requests[-1]["messages"][0]["content"]


def test_refusal_keeps_history_valid(fake_claude):
    fake_claude.replies = ["REFUSE", "Happy to help with gear!"]
    assistant = RetailAssistant(fake_claude.client(), InMemorySaver())
    thread = new_thread()
    assert assistant.ask(thread, "something off-limits")["reply"] == "Sorry, I can't help with that request."
    assert assistant.ask(thread, "ok, what tents do you have?")["reply"] == "Happy to help with gear!"
    # The refused turn added no assistant message, so the history is just the two customer messages.
    assert [m["role"] for m in fake_claude.requests[-1]["messages"]] == ["user", "user"]


def test_tool_errors_are_reported_to_claude(fake_claude):
    fake_claude.replies = [{"name": "no_such_tool", "input": {}}, "Sorry about that."]
    assistant = RetailAssistant(fake_claude.client(), InMemorySaver())
    assert assistant.ask(new_thread(), "hi")["reply"] == "Sorry about that."
    result = fake_claude.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "Unknown tool" in result["content"]


def test_tool_round_limit_closes_pending_calls(fake_claude):
    call = {"name": "search_products", "input": {"query": "tent"}}
    fake_claude.replies = [call] * (MAX_TOOL_ROUNDS + 1) + ["Next turn works."]
    assistant = RetailAssistant(fake_claude.client(), InMemorySaver())
    thread = new_thread()

    reply = assistant.ask(thread, "find me everything")["reply"]
    assert "wasn't able to finish" in reply
    assert len(fake_claude.requests) == MAX_TOOL_ROUNDS + 1

    assert assistant.ask(thread, "try again")["reply"] == "Next turn works."
    messages = fake_claude.requests[-1]["messages"]
    # The last tool call got an error result before the new customer message.
    assert messages[-2]["content"][0]["is_error"] is True
