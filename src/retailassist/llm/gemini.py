"""Gemini provider (Google Gen AI SDK, generate_content API).

Works with the Gemini Developer API (set GEMINI_API_KEY) or Vertex AI (set GOOGLE_GENAI_USE_VERTEXAI=true,
GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION; on Cloud Run the service account provides credentials).

Gemini has no native citation blocks for your own documents, so policy sections are labeled with their
section IDs and Gemini is asked to cite them as [section-id]. The graph then checks every cited ID
against the policy index and turns valid ones into numbered sources.
"""

import json
import logging
import os

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from ..rag import PolicyHit
from .base import (LLMAuthError, LLMRateLimitError, LLMUnavailableError, ModelT, ModelTurn, StopReason, ToolCall,
                   ToolResult)

DEFAULT_MODEL = "gemini-3.8-flash"
log = logging.getLogger("retailassist.llm.gemini")

# The JSON Schema keywords Gemini supports (see GenerateContentConfig.response_json_schema). Others are
# dropped from what we send, and constraints like patterns are described in words instead; the server
# validates every tool call and response against the full schema anyway.
SUPPORTED_SCHEMA_KEYS = {"$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description", "enum", "items",
                         "prefixItems", "minItems", "maxItems", "minimum", "maximum", "anyOf", "oneOf", "properties",
                         "additionalProperties", "required", "propertyOrdering"}

CITATION_INSTRUCTIONS = """
Policy sections you're given are labeled with an ID, like refund-policy#refunds. When a sentence in your \
reply relies on a policy section, put that section's ID in square brackets right after the sentence, \
like this: "Refunds go back to your original payment method [refund-policy#refunds]." Only cite IDs \
that appear in the policy sections you were given, and don't cite anything else."""

BLOCKED = {types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT, types.FinishReason.BLOCKLIST,
           types.FinishReason.SPII, types.FinishReason.RECITATION}
FAILED_TOOL_CALL = {types.FinishReason.MALFORMED_FUNCTION_CALL, types.FinishReason.UNEXPECTED_TOOL_CALL,
                    types.FinishReason.TOO_MANY_TOOL_CALLS}


class GeminiLLM:
    name = "gemini"
    instructions = CITATION_INSTRUCTIONS

    def __init__(self, client: genai.Client | None = None, model: str | None = None):
        self._client = client
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            try:
                self._client = genai.Client()
            except ValueError as e:  # raised when no API key or Vertex AI project is configured
                raise LLMAuthError("No Gemini credentials found. Set GEMINI_API_KEY, or configure Vertex AI.") from e
        return self._client

    def classify(self, system: str, prompt: str, schema: type[ModelT]) -> ModelT | None:
        """Structured output: Gemini is constrained to the schema, and the reply is validated against it here."""
        with _translate_errors():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=gemini_schema(schema.model_json_schema()),
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
        try:
            return schema.model_validate_json(response.text or "")
        except ValidationError as e:
            log.warning("Gemini's %s output didn't match the schema: %s", schema.__name__, e)
            return None

    def respond(self, system: str, messages: list[dict], tools: list) -> ModelTurn:
        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=gemini_schema(t["input_schema"]))
                        for t in (tool.to_dict() for tool in tools)]
        with _translate_errors():
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content.model_validate(m) for m in messages],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=[types.Tool(function_declarations=declarations)],
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                ),
            )

        if (response.prompt_feedback and response.prompt_feedback.block_reason) or not response.candidates:
            return ModelTurn("refusal")
        candidate = response.candidates[0]
        if candidate.finish_reason in BLOCKED:
            return ModelTurn("refusal")

        parts = (candidate.content.parts if candidate.content else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        tool_calls = [ToolCall(p.function_call.id or "", p.function_call.name, dict(p.function_call.args or {}))
                      for p in parts if p.function_call]

        stop: StopReason
        if tool_calls:
            stop = "tool_use"
        elif candidate.finish_reason == types.FinishReason.MAX_TOKENS:
            stop = "max_tokens"
        elif candidate.finish_reason in FAILED_TOOL_CALL:
            stop = "error"
        else:
            stop = "end_turn"

        message = None
        if parts:
            # Keep the whole reply, including thought signatures, which Gemini needs back on later requests.
            message = candidate.content.model_dump(mode="json", exclude_none=True)
            message["role"] = "model"
        return ModelTurn(stop, text.strip(), tool_calls, message, citations=None)

    def user_message(self, text: str, context: str, policy_hits: list[PolicyHit]) -> dict:
        parts = []
        if policy_hits:
            parts.append({"text": "<policy_sections>\n" + "\n\n".join(_format_hit(h) for h in policy_hits)
                                  + "\n</policy_sections>"})
        if context:
            parts.append({"text": f"<retrieved_context>\n{context}\n</retrieved_context>"})
        parts.append({"text": text})
        return {"role": "user", "parts": parts}

    def tool_results_message(self, results: list[ToolResult]) -> dict:
        parts = []
        for result in results:
            response = {"error": result.output} if result.is_error else {"result": _tool_payload(result.output)}
            function_response = {"name": result.call.name, "response": response}
            if result.call.id:
                function_response["id"] = result.call.id
            parts.append({"function_response": function_response})
        return {"role": "user", "parts": parts}


def gemini_schema(schema):
    """A JSON schema limited to the keywords Gemini supports, with dropped constraints put into words."""
    if isinstance(schema, list):
        return [gemini_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    hints = []
    if "pattern" in schema:
        hints.append(f"must match the regular expression {schema['pattern']}")
    if "minLength" in schema:
        hints.append(f"at least {schema['minLength']} characters")
    if "maxLength" in schema:
        hints.append(f"at most {schema['maxLength']} characters")
    if "exclusiveMinimum" in schema:
        hints.append(f"greater than {schema['exclusiveMinimum']}")
    out = {}
    for key, value in schema.items():
        if key not in SUPPORTED_SCHEMA_KEYS:
            continue
        if key in {"properties", "$defs"}:
            out[key] = {name: gemini_schema(sub) for name, sub in value.items()}
        elif key in {"items", "anyOf", "oneOf", "prefixItems", "additionalProperties"}:
            out[key] = gemini_schema(value)
        else:
            out[key] = value
    if hints:
        description = out.get("description", "").rstrip(".")
        out["description"] = (f"{description}. " if description else "") + "Must be " + ", ".join(hints) + "."
        out["description"] = out["description"].replace("Must be must match", "Must match")
    return out


def _format_hit(hit: PolicyHit) -> str:
    chunk = hit.chunk
    return f'<section id="{chunk.id}" title="{chunk.title}">\n{chunk.text}\n</section>'


def _tool_payload(output):
    if isinstance(output, list):  # policy search hits
        if not output:
            return "No policy sections matched that search."
        return [{"id": h.chunk.id, "title": h.chunk.title, "text": h.chunk.text} for h in output]
    try:
        return json.loads(output)  # most tools return JSON strings; send them as structured data
    except (TypeError, ValueError):
        return output


class _translate_errors:
    """Turn Google Gen AI SDK errors into the provider-neutral ones from base.py."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        if isinstance(exc, errors.APIError):
            message = str(exc)
            if exc.code in (401, 403) or (exc.code == 400 and "API key" in message):
                raise LLMAuthError("The Gemini API key or credentials are invalid.") from exc
            if exc.code == 429:
                raise LLMRateLimitError("Gemini rate limit reached.") from exc
            raise LLMUnavailableError(f"Gemini API error: {message}") from exc
        if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__module__.startswith(("httpx", "httpcore")):
            raise LLMUnavailableError(f"Couldn't reach the Gemini API: {exc}") from exc
        return False
