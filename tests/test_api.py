import json

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from retailassist.api import _make_client, app, get_client
from retailassist.store import ORDERS

ORDER = ORDERS["ORD-5003"]
EMAIL = ORDER["customer"]["email"]
STAFF = {"X-API-Key": "test-staff-key"}


class FakeClaude:
    """Fake Anthropic API: calls get_order_status on the first request, then answers in text."""

    def __init__(self):
        self.requests = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        base = {"id": f"msg_{len(self.requests)}", "type": "message", "role": "assistant", "model": body["model"],
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
        last = body["messages"][-1]
        if last["role"] == "user" and isinstance(last["content"], str) and "order" in last["content"]:
            return httpx2.Response(200, json={**base, "stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_order_status",
                 "input": {"order_id": "ORD-5003", "email": EMAIL}}]})
        return httpx2.Response(200, json={**base, "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "Your order has shipped."}]})


@pytest.fixture
def fake_claude():
    fake = FakeClaude()
    client = anthropic.Anthropic(api_key="test", max_retries=0,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(fake.handler)))
    app.dependency_overrides[get_client] = lambda: client
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("RETAILASSIST_STAFF_API_KEY", "test-staff-key")
    return TestClient(app)


def test_health(api):
    assert api.get("/health").json() == {"status": "ok", "products": 30, "orders": 100}


def test_chat_runs_tools_and_keeps_session(api, fake_claude):
    first = api.post("/chat", json={"message": "Where is my order?"})
    assert first.status_code == 200
    assert first.json()["reply"] == "Your order has shipped."
    # The tool result was sent back to Claude.
    tool_result = fake_claude.requests[1]["messages"][2]["content"][0]
    assert json.loads(tool_result["content"])["order_id"] == "ORD-5003"

    session_id = first.json()["session_id"]
    second = api.post("/chat", json={"message": "Thanks!", "session_id": session_id})
    assert second.json()["session_id"] == session_id
    # The second turn carries the whole first turn: user, tool call, tool result, reply, new message.
    assert len(fake_claude.requests[-1]["messages"]) == 5


def test_chat_rejects_unknown_session_and_empty_message(api, fake_claude):
    assert api.post("/chat", json={"message": "hi", "session_id": "nope"}).status_code == 404
    assert api.post("/chat", json={"message": ""}).status_code == 422


def test_chat_maps_api_errors(api):
    def fail(request):
        return httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    client = anthropic.Anthropic(api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(fail)))
    app.dependency_overrides[get_client] = lambda: client
    try:
        assert api.post("/chat", json={"message": "hi"}).status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_get_order_requires_matching_email(api):
    response = api.get("/orders/ord-5003", params={"email": EMAIL})
    assert response.status_code == 200
    assert response.json()["order_id"] == "ORD-5003"
    assert api.get("/orders/ORD-5003", params={"email": "someone@example.com"}).status_code == 404
    assert api.get("/orders/ORD-5003").status_code == 422


def test_ticket_lifecycle(api):
    created = api.post("/tickets", json={"email": EMAIL, "category": "return", "message": "Boots too small",
                                         "order_id": "ord-5003"})
    assert created.status_code == 201
    ticket = created.json()
    assert ticket["order_id"] == "ORD-5003"
    assert ticket["status"] == "open"
    assert ticket["ticket_id"].startswith("TCK-")

    assert ticket["id"] in [t["id"] for t in api.get("/tickets", headers=STAFF).json()]
    updated = api.patch(f"/tickets/{ticket['id']}", json={"status": "closed"}, headers=STAFF)
    assert updated.json()["status"] == "closed"
    assert api.get(f"/tickets/{ticket['id']}", headers=STAFF).json()["status"] == "closed"
    assert ticket["id"] in [t["id"] for t in api.get("/tickets", params={"status": "closed"}, headers=STAFF).json()]


def test_ticket_validation(api):
    assert api.post("/tickets", json={"email": "not-an-email", "category": "return", "message": "x"}).status_code == 422
    assert api.post("/tickets", json={"email": EMAIL, "category": "bogus", "message": "x"}).status_code == 422
    other = api.post("/tickets", json={"email": "someone@example.com", "category": "return", "message": "x",
                                       "order_id": "ORD-5003"})
    assert other.status_code == 422


def test_staff_endpoints_need_key(api, monkeypatch):
    assert api.get("/tickets").status_code == 401
    assert api.get("/tickets", headers={"X-API-Key": "wrong"}).status_code == 401
    assert api.get("/tickets/1").status_code == 401
    monkeypatch.delenv("RETAILASSIST_STAFF_API_KEY")
    assert api.get("/tickets", headers=STAFF).status_code == 503


def test_chat_without_credentials(api, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    _make_client.cache_clear()
    try:
        response = api.post("/chat", json={"message": "hi"})
    finally:
        _make_client.cache_clear()
    assert response.status_code == 500
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
