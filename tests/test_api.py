import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from retailassist.api import app, get_llm
from retailassist.llm.claude import ClaudeLLM
from retailassist.llm.gemini import GeminiLLM
from retailassist.store import ORDERS

ORDER = ORDERS["ORD-5003"]
EMAIL = ORDER["customer"]["email"]
STAFF = {"X-API-Key": "test-staff-key"}


@pytest.fixture
def fake_api_model(fake_model):
    app.dependency_overrides[get_llm] = fake_model.llm
    yield fake_model
    app.dependency_overrides.clear()


def use_llm(llm):
    app.dependency_overrides[get_llm] = lambda: llm


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("RETAILASSIST_STAFF_API_KEY", "test-staff-key")
    return TestClient(app)


def test_health(api, fake_api_model):
    assert api.get("/health").json() == {"status": "ok", "provider": fake_api_model.provider,
                                         "model": fake_api_model.llm().model, "products": 30, "orders": 100,
                                         "policy_chunks": 19}


def test_chat_runs_tools_and_keeps_session(api, fake_api_model):
    fake_api_model.classification = {"intent": "order_status", "order_id": "ORD-5003", "email": EMAIL, "product_query": None}
    fake_api_model.replies = [{"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}},
                               "Your order has shipped."]
    first = api.post("/chat", json={"message": "Where is my order?"})
    assert first.status_code == 200
    assert first.json()["reply"] == "Your order has shipped."
    assert first.json()["intent"] == "order_status"

    session_id = first.json()["session_id"]
    fake_api_model.classification = {"intent": "other", "order_id": None, "email": None, "product_query": None}
    second = api.post("/chat", json={"message": "Thanks!", "session_id": session_id})
    assert second.json()["session_id"] == session_id
    # The second turn carries the whole first turn: user, tool call, tool result, reply, new message.
    assert len(fake_api_model.messages(-1)) == 5


def test_chat_rejects_unknown_session_and_empty_message(api, fake_api_model):
    assert api.post("/chat", json={"message": "hi", "session_id": "nope"}).status_code == 404
    assert api.post("/chat", json={"message": ""}).status_code == 422


@pytest.mark.parametrize("status,expected", [(429, 503), (500, 502), (401, 500)])
def test_chat_maps_claude_errors(api, status, expected):
    error_type = {429: "rate_limit_error", 500: "api_error", 401: "authentication_error"}[status]
    def fail(request):
        return httpx2.Response(status, json={"type": "error", "error": {"type": error_type, "message": "nope"}})
    use_llm(ClaudeLLM(anthropic.Anthropic(api_key="test", max_retries=0,
                                          http_client=httpx2.Client(transport=httpx2.MockTransport(fail)))))
    try:
        assert api.post("/chat", json={"message": "hi"}).status_code == expected
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("status,expected", [(429, 503), (500, 502), (403, 500)])
def test_chat_maps_gemini_errors(api, fake_gemini, status, expected):
    fake_gemini.error = genai_errors.APIError(status, {"error": {"code": status, "message": "nope", "status": "X"}})
    use_llm(fake_gemini.llm())
    try:
        assert api.post("/chat", json={"message": "hi"}).status_code == expected
    finally:
        app.dependency_overrides.clear()


def test_chat_refuses_to_switch_providers(api, fake_gemini, fake_claude):
    use_llm(fake_gemini.llm())
    session_id = api.post("/chat", json={"message": "hi"}).json()["session_id"]
    use_llm(fake_claude.llm())
    try:
        response = api.post("/chat", json={"message": "hi", "session_id": session_id})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 409


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


def test_chat_without_gemini_credentials(api, monkeypatch):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    use_llm(GeminiLLM())
    try:
        response = api.post("/chat", json={"message": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_chat_without_claude_credentials(api, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    use_llm(ClaudeLLM())
    try:
        response = api.post("/chat", json={"message": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
