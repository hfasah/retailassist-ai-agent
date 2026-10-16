"""What the agent graph needs from a model provider.

Each provider keeps conversation messages in its own native format (they hold provider-specific data
such as Claude's thinking blocks or Gemini's thought signatures, which must be sent back unchanged),
so the graph treats messages as opaque dicts and only talks to them through this interface.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from ..rag import PolicyHit

StopReason = Literal["end_turn", "tool_use", "refusal", "max_tokens", "error"]
ModelT = TypeVar("ModelT", bound=BaseModel)
ToolOutput = str | list[PolicyHit]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    call: ToolCall
    output: ToolOutput
    is_error: bool = False


@dataclass
class ModelTurn:
    stop_reason: StopReason
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The reply to append to the conversation, in the provider's format. None when it can't be kept
    # (a refusal or an empty reply), since providers reject empty model turns.
    message: dict | None = None
    # Citations the provider attached itself, already numbered to match [n] markers in `text`.
    # None if the provider has no native citations; the graph then reads [section-id] markers instead.
    citations: list[dict] | None = None


class LLMError(Exception):
    """Base class for provider errors the API and CLI know how to report."""


class LLMAuthError(LLMError):
    """Missing or invalid credentials."""


class LLMRateLimitError(LLMError):
    """The provider is rate limiting us; try again shortly."""


class LLMUnavailableError(LLMError):
    """The provider couldn't be reached or returned a server error."""


class LLM(Protocol):
    name: str           # "claude" or "gemini"
    model: str
    # Extra system prompt text for this provider (e.g. how to cite sources).
    instructions: str

    def classify(self, system: str, prompt: str, schema: type[ModelT], ) -> ModelT | None: ...

    def respond(self, system: str, messages: list[dict], tools: list) -> ModelTurn: ...

    def user_message(self, text: str, context: str, policy_hits: list[PolicyHit]) -> dict: ...

    def tool_results_message(self, results: list[ToolResult]) -> dict: ...
