"""Claude provider (Anthropic API).

Policy sections are sent as `search_result` blocks, so the API attaches citations to the exact
passages Claude used.
"""

import logging
import os

import anthropic
from pydantic import ValidationError

from ..observability import get_observability
from ..rag import PolicyHit
from .base import LLMAuthError, LLMRateLimitError, LLMUnavailableError, ModelT, ModelTurn, ToolCall, ToolResult

DEFAULT_MODEL = "claude-opus-5-5"
log = logging.getLogger("retailassist.llm.claude")
# If a safety classifier declines, retry on Anthropic's recommended fallback model.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


class ClaudeLLM:
    name = "claude"
    instructions = ""  # citations are native; no extra prompting needed

    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        self._client = client
        self.model = model or os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()
            except anthropic.CredentialsError as e:
                raise LLMAuthError("Anthropic credentials couldn't be loaded. Set ANTHROPIC_API_KEY.") from e
        return self._client

    def classify(self, system: str, prompt: str, schema: type[ModelT]) -> ModelT | None:
        """Structured output: Claude is constrained to the schema, and the SDK validates the reply against it."""
        with get_observability().generation("classify", model=self.model, input=prompt,
                                            model_parameters={"effort": "low", "max_tokens": 4000}) as generation:
            try:
                with _translate_errors():
                    response = self.client.beta.messages.parse(
                        model=self.model,
                        max_tokens=4000,
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                        output_format=schema,
                        output_config={"effort": "low"},
                        **FALLBACK,
                    )
            except ValidationError as e:  # e.g. a refusal or a truncated reply that isn't valid output
                log.warning("Claude's %s output didn't match the schema: %s", schema.__name__, e)
                generation.record(usage=None, level="WARNING", status_message=f"Output didn't match {schema.__name__}")
                return None
            parsed = response.parsed_output if isinstance(response.parsed_output, schema) else None
            generation.record(usage=usage_details(response.usage), output=parsed.model_dump() if parsed else None,
                              level=None if parsed else "WARNING",
                              status_message=None if parsed else f"No valid {schema.__name__}")
            return parsed

    def respond(self, system: str, messages: list[dict], tools: list) -> ModelTurn:
        with get_observability().generation("call_model", model=self.model, input=messages,
                                            model_parameters={"effort": "medium", "max_tokens": 16000}) as generation:
            with _translate_errors():
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=system,
                    tools=[tool.to_dict() for tool in tools],
                    messages=messages,
                    output_config={"effort": "medium"},
                    **FALLBACK,
                )
            turn = self._turn(response)
            problem = {"refusal": "Refused", "max_tokens": "Hit max tokens"}
            # Claude's fallback may answer with a different model; price the call by the one that did.
            generation.model = response.model or self.model
            generation.record(usage=usage_details(response.usage),
                              output={"stop_reason": turn.stop_reason, "text": turn.text,
                                      "tool_calls": [{"name": c.name, "args": c.args} for c in turn.tool_calls]},
                              level="WARNING" if turn.stop_reason in problem else None,
                              status_message=problem.get(turn.stop_reason))
            return turn

    @staticmethod
    def _turn(response) -> ModelTurn:
        text, citations = _text_with_citations(response.content)
        tool_calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        stop = response.stop_reason if response.stop_reason in {"end_turn", "tool_use", "refusal", "max_tokens"} else "end_turn"
        message = None
        # A refused or empty reply can't go back into the history; the API rejects empty assistant turns.
        if response.stop_reason != "refusal" and response.content:
            # Plain dicts, exactly as the API returned them, so the checkpointer can save them.
            content = [block.model_dump(mode="json", exclude_unset=True) for block in response.content]
            message = {"role": "assistant", "content": content}
        return ModelTurn(stop, text, tool_calls, message, citations)

    def user_message(self, text: str, context: str, policy_hits: list[PolicyHit]) -> dict:
        content: list[dict] = [hit.to_search_result() for hit in policy_hits]
        if context:
            content.append({"type": "text", "text": f"<retrieved_context>\n{context}\n</retrieved_context>"})
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def tool_results_message(self, results: list[ToolResult]) -> dict:
        blocks = []
        for result in results:
            block = {"type": "tool_result", "tool_use_id": result.call.id, "content": _tool_content(result.output)}
            if result.is_error:
                block["is_error"] = True
            blocks.append(block)
        return {"role": "user", "content": blocks}


def usage_details(usage) -> dict[str, int] | None:
    """Token counts in Langfuse's usage format (cache reads and writes are priced separately)."""
    if usage is None:
        return None
    details = {"input": usage.input_tokens or 0, "output": usage.output_tokens or 0}
    if getattr(usage, "cache_read_input_tokens", None):
        details["input_cache_read"] = usage.cache_read_input_tokens
    if getattr(usage, "cache_creation_input_tokens", None):
        details["input_cache_creation"] = usage.cache_creation_input_tokens
    details["total"] = sum(details.values())
    return details


def _tool_content(output) -> str | list[dict]:
    if isinstance(output, str):
        return output
    if not output:
        return [{"type": "text", "text": "No policy sections matched that search."}]
    return [hit.to_search_result() for hit in output]


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


class _translate_errors:
    """Turn Anthropic SDK errors into the provider-neutral ones from base.py."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            raise LLMAuthError("The Anthropic API key is invalid.") from exc
        if isinstance(exc, anthropic.CredentialsError):
            raise LLMAuthError("Anthropic credentials couldn't be loaded. Set ANTHROPIC_API_KEY.") from exc
        # The SDK raises TypeError before sending anything when no credentials are configured.
        if isinstance(exc, TypeError) and "authentication method" in str(exc):
            raise LLMAuthError("No Anthropic API key found. Set ANTHROPIC_API_KEY.") from exc
        if isinstance(exc, anthropic.RateLimitError):
            raise LLMRateLimitError("Anthropic rate limit reached.") from exc
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APIStatusError)):
            raise LLMUnavailableError(f"Anthropic API error: {exc}") from exc
        return False

