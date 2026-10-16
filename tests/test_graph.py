import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from retailassist.graph import (MAX_TOOL_ROUNDS, ProviderMismatchError, RetailAssistant, citations_from_markers,
                                search_catalog, sqlite_checkpointer)
from retailassist.rag import get_policy_index
from retailassist.store import ORDERS

EMAIL = ORDERS["ORD-5003"]["customer"]["email"]


def new_thread() -> str:
    return uuid.uuid4().hex


def context_text(fake, request: int, message: int) -> str:
    return next(t for t in fake.texts(request, message) if "<retrieved_context>" in t)


def test_search_catalog_ranks_matches():
    assert search_catalog("waterproof hiking boots")[0]["sku"] == "SKU-1001"


# --- Behaviour shared by both providers -----------------------------------------------------------

def test_order_turn_runs_every_node(fake_model):
    fake_model.classification = {"intent": "order_status", "order_id": "ord-5003", "email": EMAIL, "product_query": None}
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}},
                          "It's on its way!"]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())

    result = assistant.ask(new_thread(), "Where is ORD-5003? I'm " + EMAIL)

    assert result["reply"] == "It's on its way!"
    assert result["intent"] == "order_status"
    # retrieve: the verified order was attached next to the customer's message.
    assert '"order_id": "ORD-5003"' in context_text(fake_model, 0, 0)
    assert fake_model.texts(0, 0)[-1].startswith("Where is ORD-5003?")
    # tool_call: the tool ran and its result went back to the model.
    output, is_error = fake_model.tool_result(1)
    assert not is_error and output["order_id"] == "ORD-5003"


def test_product_question_retrieves_products(fake_model):
    fake_model.classification = {"intent": "product_question", "order_id": None, "email": None, "product_query": "rain jacket"}
    RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "Got anything for rainy hikes?")
    assert "Stormshield Rain Jacket" in context_text(fake_model, 0, 0)


def test_conversation_persists_across_restarts(fake_model, tmp_path):
    db = tmp_path / "conversations.db"
    thread = new_thread()

    fake_model.classification = {"intent": "order_status", "order_id": "ORD-5003", "email": EMAIL, "product_query": None}
    RetailAssistant(fake_model.llm(), sqlite_checkpointer(db)).ask(thread, "Where's ORD-5003? Email " + EMAIL)

    # A new assistant with a new database connection, like after a server restart.
    fake_model.classification = {"intent": "support_request", "order_id": None, "email": None, "product_query": None}
    second = RetailAssistant(fake_model.llm(), sqlite_checkpointer(db))
    assert second.has_conversation(thread)
    assert not second.has_conversation(new_thread())
    second.ask(thread, "I want to return it")

    assert fake_model.roles(-1) == ["user", "assistant", "user"]
    # The order number and email from turn one were remembered, so turn two still retrieved the order.
    assert '"order_id": "ORD-5003"' in context_text(fake_model, -1, -1)


def test_refusal_keeps_history_valid(fake_model):
    fake_model.replies = ["REFUSE", "Happy to help with gear!"]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    assert assistant.ask(thread, "something off-limits")["reply"] == "Sorry, I can't help with that request."
    assert assistant.ask(thread, "ok, what tents do you have?")["reply"] == "Happy to help with gear!"
    # The refused turn added no model message, so the history is just the two customer messages.
    assert fake_model.roles(-1) == ["user", "user"]


def test_tool_errors_are_reported_to_the_model(fake_model):
    fake_model.replies = [{"name": "no_such_tool", "input": {}}, "Sorry about that."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    assert assistant.ask(new_thread(), "hi")["reply"] == "Sorry about that."
    output, is_error = fake_model.tool_result(1)
    assert is_error and "Unknown tool" in output


def test_tool_round_limit_closes_pending_calls(fake_model):
    call = {"name": "search_products", "input": {"query": "tent"}}
    fake_model.replies = [call] * (MAX_TOOL_ROUNDS + 1) + ["Next turn works."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()

    assert "wasn't able to finish" in assistant.ask(thread, "find me everything")["reply"]
    assert len(fake_model.requests) == MAX_TOOL_ROUNDS + 1

    assert assistant.ask(thread, "try again")["reply"] == "Next turn works."
    # The last tool call got an error result before the new customer message.
    output, is_error = fake_model.tool_result(-1, -2)
    assert is_error and "tool limit" in output


def test_policy_search_tool_results_reach_the_model(fake_model):
    fake_model.replies = [{"name": "search_store_policies", "input": {"query": "free shipping"}}, "Yes, over $75."]
    RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "hi")
    output, _ = fake_model.tool_result(1)
    titles = [r["title"] for r in output]
    assert "Shipping FAQ > What are the shipping options and costs?" in titles


def test_conversation_cant_switch_providers(fake_claude, fake_gemini):
    saver = InMemorySaver()
    thread = new_thread()
    RetailAssistant(fake_gemini.llm(), saver).ask(thread, "hi")
    with pytest.raises(ProviderMismatchError):
        RetailAssistant(fake_claude.llm(), saver).ask(thread, "hi again")


# --- Claude: native citations ---------------------------------------------------------------------

def test_claude_policy_question_retrieves_and_cites(fake_claude):
    fake_claude.classification = {"intent": "policy_question", "order_id": None, "email": None, "product_query": None}
    source = "data/policies/refund-policy.md#refunds"
    fake_claude.replies = [[
        {"type": "text", "text": "Refunds go back to your original payment method", "citations": [{
            "type": "search_result_location", "source": source, "title": "Returns & Refund Policy > Refunds",
            "cited_text": "- Refunds are issued to the **original payment method**...", "search_result_index": 0,
            "start_block_index": 0, "end_block_index": 1}]},
        {"type": "text", "text": " and usually show up within 5–10 business days."},
    ]]
    result = RetailAssistant(fake_claude.llm(), InMemorySaver()).ask(new_thread(), "how long until my money comes back?")

    # retrieve: the matching policy section went in as a citable search_result block.
    content = fake_claude.messages(0)[0]["content"]
    assert content[0]["type"] == "search_result" and content[0]["source"] == source
    assert content[0]["citations"] == {"enabled": True}
    # respond: the reply carries a numbered marker and the matching source.
    assert result["reply"] == "Refunds go back to your original payment method[1] and usually show up within 5–10 business days."
    assert result["citations"] == [{"number": 1, "title": "Returns & Refund Policy > Refunds", "source": source,
                                    "quotes": ["- Refunds are issued to the **original payment method**..."]}]


def test_claude_policy_tool_returns_search_result_blocks(fake_claude):
    fake_claude.replies = [{"name": "search_store_policies", "input": {"query": "free shipping"}}, "ok"]
    RetailAssistant(fake_claude.llm(), InMemorySaver()).ask(new_thread(), "hi")
    assert fake_claude.messages(1)[-1]["content"][0]["content"][0]["type"] == "search_result"


# --- Gemini: section-ID citations, thought signatures ---------------------------------------------

def test_gemini_policy_question_retrieves_and_cites(fake_gemini):
    fake_gemini.classification = {"intent": "policy_question", "order_id": None, "email": None, "product_query": None}
    fake_gemini.replies = ["Refunds go back to your original payment method [refund-policy#refunds]. "
                           "Return shipping is $7.95 [refund-policy#return-shipping-cost, refund-policy#refunds]. "
                           "Made-up source [refund-policy#not-a-real-section]."]
    result = RetailAssistant(fake_gemini.llm(), InMemorySaver()).ask(new_thread(), "how long until my money comes back?")

    # retrieve: policy sections are labeled with their IDs so Gemini can cite them.
    assert '<section id="refund-policy#refunds"' in fake_gemini.texts(0, 0)[0]
    # The system prompt tells Gemini how to cite.
    assert "square brackets" in fake_gemini.requests[0]["config"].system_instruction
    # respond: valid IDs become numbered sources; the made-up one is dropped.
    assert result["reply"] == ("Refunds go back to your original payment method[1]. "
                               "Return shipping is $7.95[2][1]. Made-up source.")
    assert [(c["number"], c["source"]) for c in result["citations"]] == [
        (1, "data/policies/refund-policy.md#refunds"), (2, "data/policies/refund-policy.md#return-shipping-cost")]


def test_gemini_sends_thought_signatures_and_call_ids_back(fake_gemini):
    fake_gemini.replies = [{"name": "search_products", "input": {"query": "tent"}}, "Here you go."]
    RetailAssistant(fake_gemini.llm(), InMemorySaver()).ask(new_thread(), "tents?")
    model_turn, tool_turn = fake_gemini.messages(1)[1:3]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["thought_signature"] == "c2lnbmF0dXJl"
    assert tool_turn["parts"][0]["function_response"]["id"] == "call_1"


def test_gemini_tools_are_declared_with_json_schema(fake_gemini):
    RetailAssistant(fake_gemini.llm(), InMemorySaver()).ask(new_thread(), "hi")
    declarations = fake_gemini.requests[0]["config"].tools[0].function_declarations
    names = {d.name for d in declarations}
    assert names == {"search_products", "get_product_details", "get_order_status", "search_store_policies",
                     "create_support_ticket"}
    ticket = next(d for d in declarations if d.name == "create_support_ticket")
    assert "return" in str(ticket.parameters_json_schema)


def test_citations_from_markers_ignores_unknown_ids():
    text, citations = citations_from_markers("No sources here [faq#nope].", get_policy_index())
    assert text == "No sources here." and citations == []
