"""Langfuse tracing, checked against an in-memory OpenTelemetry exporter (nothing is sent anywhere)."""

import json
import logging
import uuid

import pytest
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from retailassist.approvals import approval_store
from retailassist.graph import RetailAssistant
from retailassist.llm import LLMUnavailableError
from retailassist.observability import Observability, from_env, get_observability, mask_emails, set_observability
from retailassist.store import ORDERS

EMAIL = ORDERS["ORD-5003"]["customer"]["email"]
PRICES = {"gemini-3.8-flash": {"input": 0.30, "input_cache_read": 0.03, "output": 2.50},
          "claude-opus-5-5": {"input": 5.0, "output": 25.0}}


class Traces:
    def __init__(self):
        self.exporter = InMemorySpanExporter()
        client = Langfuse(public_key="pk-test", secret_key="sk-test", base_url="http://127.0.0.1:9",
                          span_exporter=self.exporter, tracer_provider=TracerProvider(), mask=mask_emails)
        self.obs = Observability(client, prices=PRICES)

    def spans(self) -> list:
        self.obs.flush()
        return list(self.exporter.get_finished_spans())

    def named(self, name: str) -> list:
        return [s for s in self.spans() if s.name == name]

    @staticmethod
    def attr(span, key: str):
        value = span.attributes.get(f"langfuse.observation.{key}")
        return json.loads(value) if isinstance(value, str) and value[:1] in "{[" else value


@pytest.fixture(scope="module")
def _traces():
    # One client for the module: Langfuse keeps a single client per public key.
    return Traces()


@pytest.fixture
def traces(_traces):
    previous = get_observability()
    _traces.obs.flush()
    _traces.exporter.clear()
    set_observability(_traces.obs)
    yield _traces
    set_observability(previous)


def test_tracing_is_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not from_env().enabled


def test_a_turn_is_one_trace_with_nodes_tools_and_generations(fake_model, traces):
    fake_model.classification = {"intent": "order_status", "order_id": "ORD-5003", "email": EMAIL, "product_query": None}
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": "ORD-5003", "email": EMAIL}}, "Shipped!"]
    thread = uuid.uuid4().hex
    result = RetailAssistant(fake_model.llm(), InMemorySaver()).ask(thread, f"Where is ORD-5003? I'm {EMAIL}")

    spans = traces.spans()
    assert {s.context.trace_id for s in spans} == {int(result["trace_id"], 16)}
    names = [s.name for s in spans]
    for name in ["chat turn", "start_turn", "classify", "retrieve", "call_model", "review_tool_calls", "tool_call",
                 "get_order_status", "respond"]:
        assert name in names
    [root] = traces.named("chat turn")
    assert root.parent is None and traces.attr(root, "type") == "agent"
    assert all(s.attributes["session.id"] == thread for s in spans)
    assert fake_model.provider in root.attributes["langfuse.trace.tags"]
    assert traces.attr(root, "output")["reply"] == "Shipped!"
    # The tool span sits under the tool_call node and has the validated input.
    [tool] = traces.named("get_order_status")
    [tool_node] = traces.named("tool_call")
    assert traces.attr(tool, "type") == "tool" and tool.parent.span_id == tool_node.context.span_id
    assert traces.attr(tool, "input")["order_id"] == "ORD-5003"
    # Customer emails never leave the process.
    assert all(EMAIL not in json.dumps(dict(s.attributes), default=str) for s in spans)


def test_generations_carry_model_tokens_and_cost(fake_model, traces):
    RetailAssistant(fake_model.llm(), InMemorySaver()).ask(uuid.uuid4().hex, "hi")
    generations = [s for s in traces.spans() if traces.attr(s, "type") == "generation"]
    assert {s.name for s in generations} == {"classify", "call_model"}
    generation = next(s for s in generations if s.name == "call_model")
    model = fake_model.llm().model
    assert traces.attr(generation, "model.name") == model
    usage, cost = traces.attr(generation, "usage_details"), traces.attr(generation, "cost_details")
    if fake_model.provider == "gemini":
        # 100 prompt tokens, 40 of them cached; 20 output plus 5 thinking tokens, billed as output.
        assert usage == {"input": 60, "input_cache_read": 40, "output": 25, "total": 125}
        assert cost["total"] == pytest.approx((60 * 0.30 + 40 * 0.03 + 25 * 2.50) / 1e6)
    else:
        assert usage == {"input": 10, "output": 5, "total": 15}
        assert cost["total"] == pytest.approx((10 * 5.0 + 5 * 25.0) / 1e6)
    assert traces.attr(generation, "model.parameters")


def test_turn_totals_are_logged(fake_model, traces, caplog):
    with caplog.at_level(logging.INFO, logger="retailassist.observability"):
        RetailAssistant(fake_model.llm(), InMemorySaver()).ask(uuid.uuid4().hex, "hi")
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("chat turn")]
    assert "latency_ms=" in line and "model_calls=2" in line and "cost_usd=" in line
    tokens = 125 if fake_model.provider == "gemini" else 15
    [root] = traces.named("chat turn")
    metrics = json.loads(root.attributes["langfuse.observation.metadata.metrics"])
    assert metrics["model_calls"] == 2 and metrics["total_tokens"] == 2 * tokens and metrics["cost_usd"] > 0


def test_tool_problems_are_warnings(fake_model, traces):
    fake_model.replies = [{"name": "get_order_status", "input": {"order_id": "5003", "email": EMAIL}}, "Hmm."]
    RetailAssistant(fake_model.llm(), InMemorySaver()).ask(uuid.uuid4().hex, "hi")
    [review] = traces.named("review_tool_calls")
    [tool] = traces.named("get_order_status")
    for span in (review, tool):
        assert traces.attr(span, "level") == "WARNING"
        assert "should look like ORD-5001" in traces.attr(span, "status_message")


def test_model_errors_are_captured(fake_gemini, traces):
    from google.genai import errors as genai_errors
    fake_gemini.error = genai_errors.APIError(500, {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}})
    with pytest.raises(LLMUnavailableError):
        RetailAssistant(fake_gemini.llm(), InMemorySaver()).ask(uuid.uuid4().hex, "hi")
    failed = {s.name for s in traces.spans() if traces.attr(s, "level") == "ERROR"}
    assert {"chat turn", "classify"} <= failed  # the generation, its node, and the turn
    [root] = traces.named("chat turn")
    assert "LLMUnavailableError" in traces.attr(root, "status_message")


def test_approval_pause_is_not_an_error_and_resume_is_traced(fake_model, traces):
    fake_model.replies = [{"name": "create_support_ticket", "input": {
        "email": EMAIL, "category": "cancellation", "summary": "Cancel", "order_id": "ORD-5003"}}, "Cancelled."]
    assistant = RetailAssistant(fake_model.llm(), InMemorySaver())
    thread = uuid.uuid4().hex
    request = assistant.ask(thread, "cancel")["pending_approvals"][0]
    [gate] = traces.named("approval_gate")
    assert traces.attr(gate, "level") != "ERROR" and traces.attr(gate, "status_message") == "Paused for approval"

    approval_store.decide(int(request["request_id"].removeprefix("APR-")), "approved")
    assistant.resume(thread)
    [resume] = traces.named("approval resume")
    assert resume.attributes["session.id"] == thread
    assert traces.attr(resume, "output")["reply"] == "Cancelled."
    assert "create_support_ticket" in [s.name for s in traces.spans()]


@pytest.mark.anyio
async def test_mcp_tool_calls_are_traced(traces):
    from mcp import Client
    from retailassist.mcp_server import mcp
    async with Client(mcp) as client:
        await client.call_tool("lookup_order", {"order_id": "ORD-5003", "email": EMAIL})
        await client.call_tool("lookup_order", {"order_id": "ORD-5003", "email": "x@example.com"})
    ok, refused = traces.named("mcp lookup_order")
    assert traces.attr(ok, "output")["order_id"] == "ORD-5003"
    assert traces.attr(refused, "level") == "WARNING"


@pytest.fixture
def anyio_backend():
    return "asyncio"
