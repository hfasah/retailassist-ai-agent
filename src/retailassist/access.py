"""Who may see or act on an order.

Every order lookup, in the chat agent, the API, and the MCP server, goes through authorize_order:

- The email must be the one the order was placed with. A wrong order number and a wrong email get the
  same answer, so callers can't tell which orders exist.
- Repeated failures lock the caller out for a while (per conversation and per client IP), so order
  numbers and emails can't be guessed by brute force.
- A conversation that has verified one email can't then look up orders under a different one, so a
  single chat can't be used to probe other customers' orders.
"""

import threading
import time
from collections import defaultdict, deque

from .store import find_order

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60

NO_MATCH = "No order matches that order number and email. Ask the customer to double-check both."
LOCKED_OUT = ("Too many order numbers and emails that didn't match. For the customer's security, order "
              "lookups are paused for 15 minutes.")
OTHER_CUSTOMER = ("This conversation is already verified for a different email address. Orders placed with "
                  "another email can only be looked up in a new conversation.")


class OrderAccessError(Exception):
    """The caller may not see or act on this order. The message is safe to show the customer."""


class TooManyAttemptsError(OrderAccessError):
    """Locked out after too many failed attempts."""


class FailedAttemptLimiter:
    """Counts failed order lookups per key (a conversation, a client IP) over a sliding window.

    In memory, so it's per server process; behind several instances, put a shared store in front.
    """

    def __init__(self, max_failures: int = MAX_FAILED_ATTEMPTS, window_seconds: float = LOCKOUT_SECONDS):
        self.max_failures = max_failures
        self.window = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self.max_failures

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._recent(key).append(time.monotonic())

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()

    def _recent(self, key: str) -> deque[float]:
        failures = self._failures[key]
        cutoff = time.monotonic() - self.window
        while failures and failures[0] < cutoff:
            failures.popleft()
        return failures


limiter = FailedAttemptLimiter()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def authorize_order(order_id: str, email: str, *, clients: tuple[str, ...] = (),
                    verified_email: str | None = None) -> dict:
    """Return the order if this caller may see it; otherwise raise OrderAccessError.

    clients: keys to rate-limit failures under, e.g. ("session:abc", "ip:203.0.113.7").
    verified_email: the email this conversation has already verified, if any.
    """
    if any(limiter.blocked(key) for key in clients):
        raise TooManyAttemptsError(LOCKED_OUT)
    if verified_email and normalize_email(email) != verified_email:
        _fail(clients)
        raise OrderAccessError(OTHER_CUSTOMER)
    order = find_order(order_id, email)
    if order is None:
        _fail(clients)
        raise OrderAccessError(NO_MATCH)
    return order


def _fail(clients: tuple[str, ...]) -> None:
    for key in clients:
        limiter.record_failure(key)
