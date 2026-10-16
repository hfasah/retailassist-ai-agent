import json
import os
import tempfile
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from google.genai import types as gtypes

# Keep test data out of data/. Must be set before retailassist modules are imported.
_tmp = tempfile.mkdtemp()
os.environ["RETAILASSIST_TICKETS_DB"] = os.path.join(_tmp, "tickets.db")
os.environ["RETAILASSIST_APPROVALS_DB"] = os.path.join(_tmp, "approvals.db")
os.environ["RETAILASSIST_CONVERSATIONS_DB"] = os.path.join(_tmp, "conversations.db")
os.environ["RETAILASSIST_VECTOR_DB"] = os.path.join(_tmp, "vectordb")

from retailassist.access import limiter  # noqa: E402
from retailassist.llm.claude import ClaudeLLM  # noqa: E402
from retailassist.llm.gemini import GeminiLLM  # noqa: E402

EMPTY_CLASSIFICATION = {"intent": "other", "order_id": None, "email": None, "product_query": None}


class FakeClaude:
    """A fake Anthropic API.

    Classification requests (the ones with a structured output format) get `classification` back as JSON
    (a string is sent as is, to test invalid output).
    Other requests get the next item from `replies`: a string becomes a text answer, a dict becomes a
    tool call ({"name": ..., "input": ...}), a list becomes the response's content blocks as given, and
    "REFUSE" becomes a refusal. When `replies` runs out, the answer is "Done.".
    """

    provider = "claude"

    def __init__(self):
        self.classification = dict(EMPTY_CLASSIFICATION)
        self.replies: list = []
        self.requests: list[dict] = []        # main model requests (raw JSON bodies)
        self.classify_requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        base = {"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"],
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
        if "format" in body.get("output_config", {}):
            self.classify_requests.append(body)
            text = self.classification if isinstance(self.classification, str) else json.dumps(self.classification)
            return httpx2.Response(200, json={**base, "stop_reason": "end_turn", "content": [{"type": "text", "text": text}]})

        self.requests.append(body)
        reply = self.replies.pop(0) if self.replies else "Done."
        if reply == "REFUSE":
            return httpx2.Response(200, json={**base, "stop_reason": "refusal", "content": [],
                                              "stop_details": {"type": "refusal", "category": None, "explanation": None}})
        if isinstance(reply, list):
            return httpx2.Response(200, json={**base, "stop_reason": "end_turn", "content": reply})
        if isinstance(reply, dict):
            block = {"type": "tool_use", "id": f"toolu_{len(self.requests)}", "name": reply["name"], "input": reply["input"]}
            return httpx2.Response(200, json={**base, "stop_reason": "tool_use", "content": [block]})
        return httpx2.Response(200, json={**base, "stop_reason": "end_turn", "content": [{"type": "text", "text": reply}]})

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key="test", max_retries=0,
                                   http_client=httpx2.Client(transport=httpx2.MockTransport(self.handler)))

    def llm(self) -> ClaudeLLM:
        return ClaudeLLM(self.client())

    # Provider-neutral views of what was sent, for tests that run against both providers.
    def messages(self, request: int) -> list[dict]:
        return self.requests[request]["messages"]

    def roles(self, request: int) -> list[str]:
        return [m["role"] for m in self.messages(request)]

    def texts(self, request: int, message: int) -> list[str]:
        content = self.messages(request)[message]["content"]
        return [content] if isinstance(content, str) else [b["text"] for b in content if b.get("type") == "text"]

    def tool_result(self, request: int, message: int = -1) -> tuple[object, bool]:
        block = self.messages(request)[message]["content"][0]
        assert block["type"] == "tool_result"
        content = block["content"]
        return (json.loads(content) if isinstance(content, str) and content.startswith("{") else content,
                block.get("is_error", False))


class FakeGemini:
    """A fake Gemini client with the same reply conventions as FakeClaude.

    Tool calls carry a thought signature, so tests can check it's sent back unchanged.
    """

    provider = "gemini"

    def __init__(self):
        self.classification = dict(EMPTY_CLASSIFICATION)
        self.replies: list = []
        self.requests: list[dict] = []        # main model requests: {"contents": [...dicts], "config": ...}
        self.classify_requests: list[dict] = []
        self.error: Exception | None = None
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, *, model, contents, config):
        if self.error is not None:
            raise self.error
        if config.response_json_schema is not None:
            self.classify_requests.append({"contents": contents, "config": config})
            text = self.classification if isinstance(self.classification, str) else json.dumps(self.classification)
            return self._response([{"text": text}])

        self.requests.append({"contents": [c.model_dump(mode="json", exclude_none=True) for c in contents],
                              "config": config})
        reply = self.replies.pop(0) if self.replies else "Done."
        if reply == "REFUSE":
            return gtypes.GenerateContentResponse.model_validate(
                {"candidates": [{"finish_reason": "SAFETY", "content": {"role": "model", "parts": []}}]})
        if isinstance(reply, list):
            return self._response(reply)
        if isinstance(reply, dict):
            call = {"id": f"call_{len(self.requests)}", "name": reply["name"], "args": reply["input"]}
            return self._response([{"function_call": call, "thought_signature": "c2lnbmF0dXJl"}])
        return self._response([{"text": reply}])

    @staticmethod
    def _response(parts: list[dict]) -> gtypes.GenerateContentResponse:
        return gtypes.GenerateContentResponse.model_validate(
            {"candidates": [{"finish_reason": "STOP", "content": {"role": "model", "parts": parts}}]})

    def llm(self) -> GeminiLLM:
        return GeminiLLM(client=self)

    def messages(self, request: int) -> list[dict]:
        return self.requests[request]["contents"]

    def roles(self, request: int) -> list[str]:
        return ["assistant" if m["role"] == "model" else m["role"] for m in self.messages(request)]

    def texts(self, request: int, message: int) -> list[str]:
        return [p["text"] for p in self.messages(request)[message]["parts"] if "text" in p]

    def tool_result(self, request: int, message: int = -1) -> tuple[object, bool]:
        response = self.messages(request)[message]["parts"][0]["function_response"]["response"]
        if "error" in response:
            return response["error"], True
        return response["result"], False


@pytest.fixture
def fake_claude():
    return FakeClaude()


@pytest.fixture
def fake_gemini():
    return FakeGemini()


@pytest.fixture(params=["gemini", "claude"])
def fake_model(request):
    """Runs the test once against each provider."""
    return FakeGemini() if request.param == "gemini" else FakeClaude()


@pytest.fixture(autouse=True)
def reset_failed_lookups():
    """Each test starts with no failed order lookups on record."""
    limiter.reset()
    yield
    limiter.reset()
