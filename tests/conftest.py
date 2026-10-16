import json
import os
import tempfile

import anthropic
import httpx2
import pytest

# Keep test data out of data/. Must be set before retailassist modules are imported.
_tmp = tempfile.mkdtemp()
os.environ["RETAILASSIST_TICKETS_DB"] = os.path.join(_tmp, "tickets.db")
os.environ["RETAILASSIST_CONVERSATIONS_DB"] = os.path.join(_tmp, "conversations.db")
os.environ["RETAILASSIST_VECTOR_DB"] = os.path.join(_tmp, "vectordb")


class FakeClaude:
    """A fake Anthropic API.

    Classification requests (the ones with a structured output format) get `classification` back as JSON.
    Other requests get the next item from `replies`: a string becomes a text answer, a dict becomes a
    tool call ({"name": ..., "input": ...}), a list becomes the response's content blocks as given, and
    "REFUSE" becomes a refusal. When `replies` runs out,
    the answer is "Done.".
    """

    def __init__(self):
        self.classification = {"intent": "other", "order_id": None, "email": None, "product_query": None}
        self.replies: list = []
        self.requests: list[dict] = []        # main model requests
        self.classify_requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        base = {"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"],
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
        if "format" in body.get("output_config", {}):
            self.classify_requests.append(body)
            return httpx2.Response(200, json={**base, "stop_reason": "end_turn",
                                              "content": [{"type": "text", "text": json.dumps(self.classification)}]})

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


@pytest.fixture
def fake_claude():
    return FakeClaude()
