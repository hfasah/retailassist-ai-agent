import json
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from retailassist.access import MAX_FAILED_ATTEMPTS
from retailassist.approvals import approval_store
from retailassist.graph import (MAX_TOOL_ROUNDS, ApprovalPendingError, ProviderMismatchError, RetailAssistant,
                                citations_from_markers, search_catalog, sqlite_checkpointer)
from retailassist.rag import get_policy_index
from retailassist.refunds import check_refund
from retailassist.store import ORDERS
from retailassist.tickets import ticket_store

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


# --- Human approval for write actions ------------------------------------------------------------

def ticket_call(email=EMAIL, order_id="ORD-5003"):
    return {"name": "create_support_ticket", "input": {"email": email, "category": "address_change",
                                                       "summary": "Moved house", "order_id": order_id}}


def test_ticket_waits_for_approval_then_runs(fake_model):
    fake_model.replies = [ticket_call(), "Your ticket is open."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    tickets_before = len(ticket_store.list())

    paused = assistant.ask(thread, "Please change my address")
    assert "team member needs to approve" in paused["reply"]
    [request] = paused["pending_approvals"]
    assert request["title"] == "Open an address change ticket for ORD-5003" and request["status"] == "pending"
    # Nothing was written, the model hasn't been called again, and new messages wait.
    assert len(ticket_store.list()) == tickets_before
    assert len(fake_model.requests) == 1
    with pytest.raises(ApprovalPendingError):
        assistant.ask(thread, "hello?")

    approval_id = int(request["request_id"].removeprefix("APR-"))
    approval_store.decide(approval_id, "approved", "Done for you")
    resumed = assistant.resume(thread)

    assert resumed["reply"] == "Your ticket is open." and resumed["pending_approvals"] == []
    output, is_error = fake_model.tool_result(1)
    output = json.loads(output) if isinstance(output, str) else output
    assert not is_error and output["approved"] and output["ticket_id"].startswith("TCK-")
    assert output["note_from_team"] == "Done for you"
    ticket = ticket_store.get(int(output["ticket_id"].removeprefix("TCK-")))
    assert ticket["order_id"] == "ORD-5003" and request["request_id"] in ticket["message"]
    assert approval_store.get(approval_id)["result"]["ticket_id"] == output["ticket_id"]
    # The conversation carries on normally.
    assert assistant.ask(thread, "thanks")["reply"] == "Done."


def test_rejected_action_changes_nothing(fake_model):
    fake_model.replies = [ticket_call(), "Sorry, the team declined."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    request = assistant.ask(thread, "change my address")["pending_approvals"][0]
    tickets_before = len(ticket_store.list())

    approval_store.decide(int(request["request_id"].removeprefix("APR-")), "rejected", "Already shipped")
    assert assistant.resume(thread)["reply"] == "Sorry, the team declined."
    output, _ = fake_model.tool_result(1)
    output = json.loads(output) if isinstance(output, str) else output
    assert output["approved"] is False and output["note_from_team"] == "Already shipped"
    assert len(ticket_store.list()) == tickets_before


def test_paused_conversation_survives_a_restart(fake_model, tmp_path):
    db = tmp_path / "conversations.db"
    thread = new_thread()
    fake_model.replies = [ticket_call(), "Approved after the restart."]
    request = RetailAssistant(fake_model.llm(), sqlite_checkpointer(db)).ask(thread, "change address")["pending_approvals"][0]

    approval_store.decide(int(request["request_id"].removeprefix("APR-")), "approved")
    restarted = RetailAssistant(fake_model.llm(), sqlite_checkpointer(db))
    assert restarted.resume(thread)["reply"] == "Approved after the restart."


def test_resume_before_decision_stays_paused(fake_model):
    fake_model.replies = [ticket_call()]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    assistant.ask(thread, "change my address")
    assert assistant.resume(thread)["pending_approvals"]
    assert len(fake_model.requests) == 1


def test_eligible_refund_needs_approval(fake_model):
    order = next(o for o in ORDERS.values() if check_refund(o, "wrong_size").eligible)
    email = order["customer"]["email"]
    fake_model.replies = [{"name": "request_refund", "input": {"order_id": order["order_id"], "email": email,
                                                               "reason": "wrong_size"}}, "Refund approved."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()

    [request] = assistant.ask(thread, "refund please")["pending_approvals"]
    assert request["title"].startswith("Refund $") and order["order_id"] in request["title"]
    approval = approval_store.get(int(request["request_id"].removeprefix("APR-")))
    assert approval["action"] == "request_refund" and approval["source"] == "chat"
    assert ticket_store.find_open(order_id=order["order_id"], category="refund") is None

    approval_store.decide(approval["id"], "approved")
    assistant.resume(thread)
    ticket = ticket_store.find_open(order_id=order["order_id"], category="refund")
    assert ticket and ticket["source"] == "chat"
    ticket_store.update_status(ticket["id"], "closed")  # so other tests can request a refund for it again


def test_ineligible_refund_skips_approval(fake_model):
    order = next(o for o in ORDERS.values() if o["status"] == "shipped")
    fake_model.replies = [{"name": "request_refund", "input": {"order_id": order["order_id"],
                                                               "email": order["customer"]["email"],
                                                               "reason": "changed_mind"}}, "Not yet, sorry."]
    result = RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "refund please")
    assert result["reply"] == "Not yet, sorry." and result["pending_approvals"] == []
    output, is_error = fake_model.tool_result(1)
    output = json.loads(output) if isinstance(output, str) else output
    assert not is_error and output["eligible"] is False


# --- Order access ---------------------------------------------------------------------------------

def other_order(email: str) -> dict:
    return next(o for o in ORDERS.values() if o["customer"]["email"] != email)


def test_write_action_for_someone_elses_order_is_refused(fake_model):
    fake_model.replies = [ticket_call(email="someone@example.com"), "Those details don't match."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    result = assistant.ask(thread, "cancel ORD-5003")
    assert result["pending_approvals"] == [] and approval_store.list(thread_id=thread) == []
    output, is_error = fake_model.tool_result(1)
    assert is_error and "No order matches" in output


def test_conversation_is_bound_to_the_first_verified_email(fake_model):
    other = other_order(EMAIL)
    fake_model.replies = [
        {"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}}, "Found it.",
        # The right email for another customer's order, in the same conversation.
        {"name": "get_order_status", "input": {"order_id": other["order_id"], "email": other["customer"]["email"]}},
        "Can't do that.",
        ticket_call(email=other["customer"]["email"], order_id=None), "Can't do that either.",
    ]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    assistant.ask(thread, "where's ORD-5003?")
    assistant.ask(thread, f"and {other['order_id']}?")
    output, is_error = fake_model.tool_result(-1)
    assert is_error and "different email" in output
    assistant.ask(thread, "open a ticket for my friend")
    output, is_error = fake_model.tool_result(-1)
    assert is_error and "different email" in output


def test_retrieve_refuses_orders_under_another_email(fake_model):
    other = other_order(EMAIL)
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = new_thread()
    fake_model.classification = {"intent": "order_status", "order_id": "ORD-5003", "email": EMAIL, "product_query": None}
    assistant.ask(thread, "where's ORD-5003?")
    fake_model.classification = {"intent": "order_status", "order_id": other["order_id"],
                                 "email": other["customer"]["email"], "product_query": None}
    assistant.ask(thread, "what about this one?")
    context = context_text(fake_model, -1, -1)
    assert "refused" in context and other["customer"]["name"] not in context
    assert other["order_id"] in context and '"items"' not in context


def test_failed_lookups_lock_the_conversation(fake_model):
    wrong = {"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": "guess@example.com"}}
    right = {"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}}
    fake_model.replies = [wrong] * MAX_FAILED_ATTEMPTS + [right, "Locked."]
    RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "guessing")
    # Even the right email is refused once the conversation is locked.
    output, is_error = fake_model.tool_result(-1)
    assert is_error and "paused for 15 minutes" in output


def test_failed_lookups_lock_the_client_across_conversations(fake_model):
    wrong = {"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": "guess@example.com"}}
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    for _ in range(MAX_FAILED_ATTEMPTS):
        fake_model.replies = [wrong, "No."]
        assistant.ask(new_thread(), "guessing", client="203.0.113.7")
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}}, "Locked."]
    assistant.ask(new_thread(), "now the real one", client="203.0.113.7")
    output, is_error = fake_model.tool_result(-1)
    assert is_error and "paused" in output
    # Another client isn't affected.
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}}, "Here."]
    assistant.ask(new_thread(), "hi", client="198.51.100.2")
    assert not fake_model.tool_result(-1)[1]


# --- Validation -----------------------------------------------------------------------------------

def tool_output(fake, request: int):
    output, is_error = fake.tool_result(request)
    return (json.loads(output) if isinstance(output, str) and output.startswith("{") else output), is_error


def test_bad_tool_arguments_are_explained_so_the_model_can_fix_them(fake_model):
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": "5003", "email": EMAIL}},
                          {"name": "get_order_status", "input": {"order_id": "ord-5003", "email": EMAIL.upper()}},
                          "Found it."]
    assert RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "where's 5003?")["reply"] == "Found it."
    error, is_error = tool_output(fake_model, 1)
    assert is_error and "order_id: should look like ORD-5001" in error
    # The corrected (and normalized) call ran, and returned the validated output model.
    order, is_error = tool_output(fake_model, 2)
    assert not is_error and order["order_id"] == "ORD-5003" and "customer" not in order and order["customer_name"]


def test_wrongly_typed_arguments_dont_crash_the_turn(fake_model):
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": 5003, "email": None}},
                          {"name": "create_support_ticket", "input": {"email": EMAIL, "category": "refund-please",
                                                                      "summary": ""}},
                          "Sorry."]
    thread = new_thread()
    assert RetailAssistant(fake_model.llm(), InMemorySaver()).ask(thread, "hi")["reply"] == "Sorry."
    error, is_error = tool_output(fake_model, 1)
    assert is_error and "order_id: Input should be a valid string" in error and "email:" in error
    error, is_error = tool_output(fake_model, 2)
    assert is_error and "category:" in error and "summary:" in error
    assert approval_store.list(thread_id=thread) == []  # invalid write requests never reach staff


def test_classification_drops_malformed_details(fake_model):
    fake_model.classification = {"intent": "order_status", "order_id": "my order", "email": "not an email",
                                 "product_query": None}
    RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "where's my order?")
    assert "<retrieved_context>" not in "".join(fake_model.texts(0, 0))  # nothing was looked up


@pytest.mark.parametrize("bad", ["not json", '{"intent": "shopping_spree"}'])
def test_invalid_classification_falls_back_to_other(fake_model, bad):
    fake_model.classification = bad
    result = RetailAssistant(fake_model.llm(), InMemorySaver()).ask(new_thread(), "hi")
    assert result["intent"] == "other" and result["reply"] == "Done."


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


def test_gemini_gets_only_supported_schema_keywords(fake_gemini):
    RetailAssistant(fake_gemini.llm(), InMemorySaver()).ask(new_thread(), "hi")
    declarations = {d.name: d for d in fake_gemini.requests[0]["config"].tools[0].function_declarations}
    order_id = declarations["get_order_status"].parameters_json_schema["properties"]["order_id"]
    assert "pattern" not in order_id and "ORD-" in order_id["description"]
    classify_schema = fake_gemini.classify_requests[0]["config"].response_json_schema
    assert set(classify_schema["properties"]) == {"intent", "order_id", "email", "product_query"}


def test_gemini_tools_are_declared_with_json_schema(fake_gemini):
    RetailAssistant(fake_gemini.llm(), InMemorySaver()).ask(new_thread(), "hi")
    declarations = fake_gemini.requests[0]["config"].tools[0].function_declarations
    names = {d.name for d in declarations}
    assert names == {"search_products", "get_product_details", "get_order_status", "search_store_policies",
                     "create_support_ticket", "request_refund"}
    ticket = next(d for d in declarations if d.name == "create_support_ticket")
    assert "return" in str(ticket.parameters_json_schema)


def test_citations_from_markers_ignores_unknown_ids():
    text, citations = citations_from_markers("No sources here [faq#nope].", get_policy_index())
    assert text == "No sources here." and citations == []
