"""Tracing, error capture, latency, and token/cost metrics, sent to Langfuse.

Turned on by setting LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (plus LANGFUSE_BASE_URL for a
self-hosted or regional Langfuse; Langfuse Cloud EU is the default). Without them, every call here is a
cheap no-op, so the rest of the code never has to check.

Each chat turn (or approval resume) is one trace, grouped by session ID, laid out like the graph:

    chat turn (agent)                     session_id = conversation, tags = [provider, model]
    ├── classify (chain) → classify (generation: model, tokens, cost, latency)
    ├── retrieve (retriever)
    ├── call_model (chain) → call_model (generation)
    ├── review_tool_calls (guardrail)
    ├── approval_gate (guardrail)        "paused for approval" when it interrupts
    ├── tool_call (chain) → search_products, get_order_status, ... (tool)
    └── respond (chain)

- Errors: an exception in any step marks that observation (and the ones around it) ERROR with the
  message; tool errors and invalid tool arguments are WARNING, so they're easy to filter on.
- Latency: every observation is timed by Langfuse; each turn also logs its total latency.
- Tokens and cost: providers report usage on every generation (including cached and thinking tokens).
  Langfuse prices known models itself; to set prices yourself, set RETAILASSIST_MODEL_PRICES to JSON
  of USD per million tokens, e.g. {"gemini-3.8-flash": {"input": 0.3, "output": 2.5}}, and the cost is
  sent with each generation. Turn totals are logged and attached to the trace.
- Privacy: email addresses are masked before anything leaves the process (RETAILASSIST_TRACE_MASK_EMAILS=false
  to turn that off).
"""

import contextvars
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

log = logging.getLogger("retailassist.observability")

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@dataclass
class TurnMetrics:
    """Totals for one trace, logged at the end and attached to the trace's output metadata."""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    tool_calls: int = 0
    tool_errors: int = 0
    errors: list[str] = field(default_factory=list)

    def add_usage(self, usage: dict[str, int], cost: dict[str, float] | None) -> None:
        self.model_calls += 1
        self.input_tokens += sum(v for k, v in usage.items() if k.startswith("input"))
        self.output_tokens += usage.get("output", 0)
        self.total_tokens += usage.get("total", 0)
        if cost:
            self.cost_usd = round((self.cost_usd or 0.0) + cost["total"], 6)


_metrics: contextvars.ContextVar[TurnMetrics | None] = contextvars.ContextVar("retailassist_metrics", default=None)


class _NoOp:
    """Stands in for a Langfuse observation when tracing is off."""

    def update(self, **kwargs: Any) -> "_NoOp":
        return self


def mask_emails(*, data: Any, **kwargs: Any) -> Any:
    if isinstance(data, str):
        return EMAIL.sub("[email]", data)
    if isinstance(data, dict):
        return {k: mask_emails(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_emails(data=v) for v in data]
    return data


def load_prices() -> dict[str, dict[str, float]]:
    raw = os.environ.get("RETAILASSIST_MODEL_PRICES")
    if not raw:
        return {}
    try:
        prices = json.loads(raw)
        assert isinstance(prices, dict) and all(isinstance(p, dict) for p in prices.values())
        return prices
    except (ValueError, AssertionError):
        log.warning("Ignoring RETAILASSIST_MODEL_PRICES: it must be JSON like {\"model\": {\"input\": 0.3}}")
        return {}


class Observability:
    def __init__(self, client=None, prices: dict[str, dict[str, float]] | None = None):
        self.client = client  # a langfuse.Langfuse, or None when tracing is off
        self.prices = prices if prices is not None else load_prices()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def trace(self, name: str, *, session_id: str, input: Any = None, tags: list[str] | None = None,
              metadata: dict[str, str] | None = None, expected: tuple[type[Exception], ...] = ()) -> Iterator[Any]:
        """The root of one trace: a chat turn, an approval resume, or an MCP tool call. Logs its latency
        and totals. Exceptions of the `expected` types (such as "no order matches") are WARNING, not ERROR."""
        metrics = TurnMetrics()
        token = _metrics.set(metrics)
        started = time.perf_counter()
        try:
            if not self.enabled:
                yield _NoOp()
                return
            from langfuse import propagate_attributes
            with self.client.start_as_current_observation(name=name, as_type="agent", input=input) as root, \
                    propagate_attributes(session_id=session_id, trace_name=name, tags=tags, metadata=metadata):
                try:
                    yield root
                except expected as e:
                    self.warn(root, f"{type(e).__name__}: {e}")
                    raise
                except Exception as e:
                    _mark_error(root, e)
                    raise
                finally:
                    root.update(metadata={"metrics": asdict(metrics)})
        except expected:
            raise
        except Exception as e:
            metrics.errors.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            _metrics.reset(token)
            latency_ms = round((time.perf_counter() - started) * 1000)
            log.info("%s session=%s latency_ms=%d model_calls=%d tokens_in=%d tokens_out=%d cost_usd=%s tools=%d "
                     "tool_errors=%d errors=%d", name, session_id, latency_ms, metrics.model_calls,
                     metrics.input_tokens, metrics.output_tokens, metrics.cost_usd, metrics.tool_calls,
                     metrics.tool_errors, len(metrics.errors))

    @contextmanager
    def observe(self, name: str, as_type: str = "span", input: Any = None, metadata: Any = None) -> Iterator[Any]:
        """One step inside a trace: a graph node, a tool, a retrieval. Exceptions mark it ERROR."""
        if not self.enabled:
            yield _NoOp()
            return
        with self.client.start_as_current_observation(name=name, as_type=as_type, input=input,
                                                      metadata=metadata) as observation:
            try:
                yield observation
            except Exception as e:
                if _is_pause(e):
                    observation.update(status_message="Paused for approval")
                else:
                    _mark_error(observation, e)
                raise

    @contextmanager
    def generation(self, name: str, *, model: str, input: Any = None,
                   model_parameters: dict | None = None) -> Iterator["Generation"]:
        """One model call. Call .record(usage=..., output=...) on what's yielded once it returns."""
        if not self.enabled:
            yield Generation(self, None, model)
            return
        with self.client.start_as_current_observation(name=name, as_type="generation", model=model, input=input,
                                                      model_parameters=model_parameters) as observation:
            try:
                yield Generation(self, observation, model)
            except Exception as e:
                _mark_error(observation, e)
                raise

    def cost(self, model: str, usage: dict[str, int]) -> dict[str, float] | None:
        """Cost in USD per usage type from RETAILASSIST_MODEL_PRICES, or None to let Langfuse price it."""
        prices = self.prices.get(model)
        if not prices:
            return None
        cost = {key: round(count * prices[key] / 1_000_000, 8) for key, count in usage.items()
                if key != "total" and key in prices}
        cost["total"] = round(sum(cost.values()), 8)
        return cost

    def warn(self, observation: Any, message: str) -> None:
        """Flag a recoverable problem (a tool error, invalid tool arguments) on an observation."""
        observation.update(level="WARNING", status_message=message[:500])

    def count_tool(self, error: bool) -> None:
        if metrics := _metrics.get():
            metrics.tool_calls += 1
            metrics.tool_errors += int(error)

    def trace_id(self) -> str | None:
        return self.client.get_current_trace_id() if self.enabled else None

    def flush(self) -> None:
        if self.enabled:
            self.client.flush()

    def shutdown(self) -> None:
        if self.enabled:
            self.client.shutdown()


@dataclass
class Generation:
    obs: Observability
    observation: Any
    model: str

    def record(self, *, usage: dict[str, int] | None, output: Any = None, level: str | None = None,
               status_message: str | None = None) -> None:
        cost = self.obs.cost(self.model, usage) if usage else None
        if metrics := _metrics.get():
            metrics.add_usage(usage or {}, cost)
        if self.observation is not None:
            self.observation.update(output=output, usage_details=usage or None, cost_details=cost, level=level,
                                    status_message=status_message)


def _mark_error(observation: Any, error: Exception) -> None:
    # Only the trace root counts errors in TurnMetrics, so one failure isn't counted at every level.
    observation.update(level="ERROR", status_message=f"{type(error).__name__}: {error}"[:500])


def _is_pause(error: Exception) -> bool:
    """LangGraph pauses for approval by raising GraphInterrupt; that's expected, not an error."""
    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:  # pragma: no cover
        return False
    return isinstance(error, GraphInterrupt)


def from_env() -> Observability:
    """Langfuse tracing if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, otherwise a no-op."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return Observability()
    if os.environ.get("RETAILASSIST_TRACING", "on").lower() in {"off", "false", "0"}:
        return Observability()
    from langfuse import Langfuse
    mask = None if os.environ.get("RETAILASSIST_TRACE_MASK_EMAILS", "true").lower() in {"false", "0", "off"} \
        else mask_emails
    return Observability(Langfuse(mask=mask, release=os.environ.get("RETAILASSIST_RELEASE")))


_observability: Observability | None = None


def get_observability() -> Observability:
    global _observability
    if _observability is None:
        _observability = from_env()
    return _observability


def set_observability(obs: Observability) -> None:
    """Swap the global instance (tests use this with an in-memory exporter)."""
    global _observability
    _observability = obs
