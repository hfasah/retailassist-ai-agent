"""RetailAssist MCP server: order lookup, support tickets, and refund requests.

Any MCP client (Claude Desktop, Claude Code, Gemini CLI, an agent framework) can use these tools.

    retailassist-mcp                                    # stdio, for local MCP clients
    RETAILASSIST_MCP_TOKEN=... retailassist-mcp --transport http --host 0.0.0.0 --port 8080

Over HTTP every request needs "Authorization: Bearer <RETAILASSIST_MCP_TOKEN>". Customers are verified
by order number plus the email the order was placed with, the same as the rest of RetailAssist, and
repeated failures pause order lookups (access.py).

Nothing here writes on its own: create_ticket and request_refund file a request that a staff member
approves through the API (/approvals) before it runs. This server never issues money.
"""

import argparse
import logging
import os
import secrets
import sys
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .access import OrderAccessError, authorize_order, normalize_email
from .approvals import RefundArgs, TicketArgs, approval_store, propose_refund, propose_ticket
from .refunds import RefundReason
from .schemas import Details, Email, Message, Order, OrderId, RefundOutcome, Sku
from .tickets import TicketCategory

log = logging.getLogger("retailassist.mcp")

INSTRUCTIONS = """Tools for an outdoor gear store's customer support.

Every tool that touches an order needs the order number (like ORD-5001) and the email address the order \
was placed with; ask the customer for both. Use lookup_order to check status, items, and delivery dates. \
Use request_refund when a customer wants their money back: it checks the refund policy and, if the \
request qualifies, sends it to a staff member to approve. Use create_ticket for anything else that needs \
a person (cancellations, address changes, missing packages); it also waits for staff approval. Both return \
a request ID; use get_request_status to see whether it has been approved. Nothing is refunded directly."""

mcp = MCPServer("retailassist", title="RetailAssist", instructions=INSTRUCTIONS, version="0.1.0")

RequestId = Annotated[str, Field(description="Request ID from create_ticket or request_refund, e.g. APR-0001",
                                  pattern=r"^APR-\d+$")]
# Failed lookups from MCP clients are rate-limited together (access.py).
CLIENTS = ("mcp",)


# --- Output models (sent as structured content) ---------------------------------------------------

class RequestStatus(BaseModel):
    request_id: str = Field(description="Use with get_request_status")
    title: str
    status: Literal["pending", "approved", "rejected"] = Field(
        description="pending until a staff member decides; approved requests have already run")
    note: str = Field(description="Note from the staff member who decided, if any")
    result: dict | None = Field(description="What happened once approved, e.g. the ticket ID")


RefundResult = RefundOutcome


# --- Tools ----------------------------------------------------------------------------------------

def _authorize(order_id: str, email: str) -> dict:
    try:
        return authorize_order(order_id, email, clients=CLIENTS)
    except OrderAccessError as e:
        raise ToolError(str(e))


def _status(approval: dict) -> RequestStatus:
    return RequestStatus(request_id=approval["approval_id"], title=approval["title"], status=approval["status"],
                         note=approval["note"], result=approval["result"])


@mcp.tool(annotations=ToolAnnotations(title="Look up an order", read_only_hint=True, destructive_hint=False,
                                      idempotent_hint=True, open_world_hint=False))
def lookup_order(order_id: OrderId, email: Email) -> Order:
    """Look up a customer's order: status, items, totals, carrier and tracking number, and delivery dates.

    The email must match the one the order was placed with.
    """
    return Order.model_validate(_authorize(order_id, email))


@mcp.tool(annotations=ToolAnnotations(title="Request a support ticket", read_only_hint=False,
                                      destructive_hint=False, idempotent_hint=False, open_world_hint=False))
def create_ticket(
    email: Email,
    category: Annotated[TicketCategory, Field(description="The kind of request")],
    message: Annotated[Message, Field(description="What the customer needs, for the support team")],
    order_id: Annotated[OrderId | None, Field(description="Related order number, if any. Must match the email.")] = None,
) -> RequestStatus:
    """Ask for a support ticket for a team member to follow up on, such as a cancellation, address change,
    or missing package. A staff member approves it before it's opened; check with get_request_status.
    For refunds, use request_refund instead, which checks the refund policy first."""
    order = _authorize(order_id, email) if order_id else None
    proposal = propose_ticket(TicketArgs(email=email, category=category, message=message, order_id=order_id), order)
    approval = approval_store.submit(proposal, source="mcp")
    log.info("Ticket request %s (%s) via MCP, waiting for approval", approval["approval_id"], category)
    return _status(approval)


@mcp.tool(annotations=ToolAnnotations(title="Request a refund", read_only_hint=False, destructive_hint=False,
                                      idempotent_hint=True, open_world_hint=False))
def request_refund(
    order_id: OrderId,
    email: Email,
    reason: Annotated[RefundReason, Field(description="Why the customer wants a refund")],
    skus: Annotated[list[Sku] | None, Field(description="SKUs to refund, e.g. ['SKU-1001']. Omit for the whole order.",
                                            max_length=20)] = None,
    details: Annotated[Details, Field(description="Anything else the support team should know")] = "",
) -> RefundResult:
    """Check a refund request against the store's refund policy and, if it qualifies, send it to a staff
    member to approve. Returns whether it qualifies, why, and the estimated refund amount.

    This does not issue the refund. Calling it again for the same order returns the request (or refund
    ticket) that's already open instead of creating another one.
    """
    order = _authorize(order_id, email)
    outcome, proposal = propose_refund(RefundArgs(order_id=order_id, email=email, reason=reason, skus=skus,
                                                  details=details), order)
    result = RefundResult(**outcome, request_id=None)
    if proposal is None:
        return result
    approval = approval_store.find_pending("request_refund", order["order_id"])
    result.already_requested = approval is not None
    approval = approval or approval_store.submit(proposal, source="mcp")
    result.request_id = approval["approval_id"]
    result.next_step = (f"Waiting for a team member to approve {approval['approval_id']}. Check with "
                        "get_request_status; once approved, the customer gets a prepaid return label by email.")
    log.info("Refund request %s for %s via MCP", approval["approval_id"], order["order_id"])
    return result


@mcp.tool(annotations=ToolAnnotations(title="Check a request", read_only_hint=True, destructive_hint=False,
                                      idempotent_hint=True, open_world_hint=False))
def get_request_status(request_id: RequestId, email: Email) -> RequestStatus:
    """See whether a ticket or refund request has been approved, and what happened once it was.

    The email must be the customer's, as given when the request was made.
    """
    approval = approval_store.get(int(request_id.removeprefix("APR-")))
    if approval is None or approval["customer_email"] != normalize_email(email):
        raise ToolError("No request matches that ID and email.")
    return _status(approval)


# --- Transports -----------------------------------------------------------------------------------

class BearerAuth:
    """ASGI middleware: reject HTTP requests without the right bearer token."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            header = dict(scope["headers"]).get(b"authorization", b"")
            if not secrets.compare_digest(header, self.expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"error": "Missing or invalid bearer token."}'})
                return
        await self.app(scope, receive, send)


def http_app(token: str, host: str = "127.0.0.1"):
    """The MCP server as an ASGI app (stateless JSON over streamable HTTP), behind bearer-token auth."""
    return BearerAuth(mcp.streamable_http_app(stateless_http=True, json_response=True, host=host), token)


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailAssist MCP server.")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP only. Use 0.0.0.0 in a container.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8001)), help="HTTP only.")
    args = parser.parse_args()
    # stdout carries the MCP protocol over stdio, so logs go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.transport == "stdio":
        mcp.run("stdio")
        return

    token = os.environ.get("RETAILASSIST_MCP_TOKEN")
    if not token:
        parser.error("Set RETAILASSIST_MCP_TOKEN before serving over HTTP; clients must send it as a bearer token.")
    import uvicorn
    uvicorn.run(http_app(token, args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
