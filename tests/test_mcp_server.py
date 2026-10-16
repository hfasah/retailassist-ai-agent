import os
import sys

import pytest
from mcp import Client, StdioServerParameters
from starlette.testclient import TestClient

from retailassist.access import MAX_FAILED_ATTEMPTS
from retailassist.approvals import approval_store, run_action
from retailassist.mcp_server import http_app, mcp
from retailassist.refunds import check_refund
from retailassist.store import ORDERS

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def eligible_order() -> dict:
    """A delivered order with no final-sale items that qualifies for a size-related refund today."""
    return next(o for o in ORDERS.values() if check_refund(o, "wrong_size").eligible)


def ineligible_order() -> dict:
    return next(o for o in ORDERS.values() if o["status"] == "shipped")


async def test_tools_are_listed_with_annotations_and_schemas():
    async with Client(mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"lookup_order", "create_ticket", "request_refund", "get_request_status"}
    assert tools["lookup_order"].annotations.read_only_hint is True
    assert tools["get_request_status"].annotations.read_only_hint is True
    assert tools["create_ticket"].annotations.read_only_hint is False
    assert tools["request_refund"].annotations.idempotent_hint is True
    assert all(t.annotations.destructive_hint is False for t in tools.values())
    assert all(t.output_schema for t in tools.values())
    assert tools["request_refund"].input_schema["required"] == ["order_id", "email", "reason"]


async def test_lookup_order():
    order = ORDERS["ORD-5003"]
    async with Client(mcp) as client:
        found = await client.call_tool("lookup_order", {"order_id": "ord-5003", "email": order["customer"]["email"]})
        wrong = await client.call_tool("lookup_order", {"order_id": "ORD-5003", "email": "someone@example.com"})
    assert not found.is_error
    assert found.structured_content["order_id"] == "ORD-5003"
    assert "customer" not in found.structured_content  # only the order, not the stored customer record
    assert wrong.is_error and "No order matches" in wrong.content[0].text


def approve(request_id: str) -> dict:
    """What a staff member does through POST /approvals/{id}."""
    approval = approval_store.decide(int(request_id.removeprefix("APR-")), "approved", "OK")
    return run_action(approval)


async def test_create_ticket_waits_for_approval():
    from retailassist.tickets import ticket_store
    email = ORDERS["ORD-5003"]["customer"]["email"]
    async with Client(mcp) as client:
        created = await client.call_tool("create_ticket", {
            "email": email, "category": "address_change", "message": "Moved house", "order_id": "ord-5003"})
        mismatched = await client.call_tool("create_ticket", {
            "email": "someone@example.com", "category": "other", "message": "x", "order_id": "ORD-5003"})
        no_order = await client.call_tool("create_ticket", {"email": "someone@example.com", "category": "other",
                                                            "message": "General question"})
        request = created.structured_content
        assert request["status"] == "pending" and request["result"] is None
        assert request["title"] == "Open an address change ticket for ORD-5003"

        ticket_id = approve(request["request_id"])["ticket_id"]
        status = await client.call_tool("get_request_status", {"request_id": request["request_id"], "email": email})
        snooping = await client.call_tool("get_request_status", {"request_id": request["request_id"],
                                                                 "email": "someone@example.com"})

    assert mismatched.is_error and "No order matches" in mismatched.content[0].text
    assert not no_order.is_error and no_order.structured_content["status"] == "pending"
    assert status.structured_content["status"] == "approved"
    assert status.structured_content["result"]["ticket_id"] == ticket_id
    ticket = ticket_store.get(int(ticket_id.removeprefix("TCK-")))
    assert ticket["order_id"] == "ORD-5003" and ticket["source"] == "mcp"
    assert snooping.is_error


async def test_eligible_refund_waits_for_approval_and_opens_one_ticket():
    from retailassist.tickets import ticket_store
    order = eligible_order()
    args = {"order_id": order["order_id"], "email": order["customer"]["email"], "reason": "wrong_size",
            "details": "Runs small"}
    async with Client(mcp) as client:
        first = (await client.call_tool("request_refund", args)).structured_content
        second = (await client.call_tool("request_refund", args)).structured_content
        assert first["eligible"] and first["request_id"] and first["ticket_id"] is None
        assert first["estimated_refund"] > 0 and not first["already_requested"]
        # Asking again while it's waiting returns the same request instead of another one.
        assert second["already_requested"] and second["request_id"] == first["request_id"]
        assert ticket_store.find_open(order_id=order["order_id"], category="refund") is None

        ticket_id = approve(first["request_id"])["ticket_id"]
        third = (await client.call_tool("request_refund", args)).structured_content

    ticket = ticket_store.get(int(ticket_id.removeprefix("TCK-")))
    assert ticket["category"] == "refund" and ticket["source"] == "mcp"
    assert "Runs small" in ticket["message"] and f"${first['estimated_refund']:.2f}" in ticket["message"]
    # Once the refund ticket is open, asking again points to it.
    assert third["already_requested"] and third["ticket_id"] == ticket_id and third["request_id"] is None
    ticket_store.update_status(ticket["id"], "closed")


async def test_failed_lookups_are_rate_limited():
    async with Client(mcp) as client:
        for _ in range(MAX_FAILED_ATTEMPTS):
            await client.call_tool("lookup_order", {"order_id": "ORD-5003", "email": "guess@example.com"})
        locked = await client.call_tool("lookup_order", {"order_id": "ORD-5003",
                                                         "email": ORDERS["ORD-5003"]["customer"]["email"]})
    assert locked.is_error and "paused" in locked.content[0].text


async def test_ineligible_refund_opens_no_ticket():
    order = ineligible_order()
    async with Client(mcp) as client:
        result = await client.call_tool("request_refund", {"order_id": order["order_id"],
                                                           "email": order["customer"]["email"], "reason": "changed_mind"})
    assert not result.is_error
    assert result.structured_content["eligible"] is False
    assert result.structured_content["ticket_id"] is None
    assert "hasn't been delivered" in result.structured_content["reasons"][0]


async def test_invalid_arguments_are_rejected():
    async with Client(mcp) as client:
        bad_reason = await client.call_tool("request_refund", {"order_id": "ORD-5003", "email": "a@example.com",
                                                               "reason": "free money"})
        bad_email = await client.call_tool("lookup_order", {"order_id": "ORD-5003", "email": "not-an-email"})
    assert bad_reason.is_error and bad_email.is_error


def test_http_requires_bearer_token():
    # Local servers only accept localhost Host headers (DNS rebinding protection).
    with TestClient(http_app("secret"), base_url="http://127.0.0.1:8001") as http:
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        headers = {"accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"}
        assert http.post("/mcp", json=request, headers=headers).status_code == 401
        assert http.post("/mcp", json=request, headers={**headers, "authorization": "Bearer nope"}).status_code == 401
        ok = http.post("/mcp", json=request, headers={**headers, "authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert {t["name"] for t in ok.json()["result"]["tools"]} == {"lookup_order", "create_ticket", "request_refund",
                                                                 "get_request_status"}


async def test_stdio_transport():
    server = StdioServerParameters(command=sys.executable, args=["-m", "retailassist.mcp_server"], env=dict(os.environ))
    async with Client(server) as client:
        result = await client.call_tool("lookup_order", {"order_id": "ORD-5003",
                                                         "email": ORDERS["ORD-5003"]["customer"]["email"]})
    assert result.structured_content["order_id"] == "ORD-5003"
