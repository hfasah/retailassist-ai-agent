"""Tools the agent can call. Each returns a string (or policy hits) that is sent back to the model.

Read-only tools run directly. Write tools (WRITE_TOOLS) only declare what the model can ask for: the
graph checks order access, gets a person's approval, and then runs the action (see graph.py and
approvals.py). Any tool that takes an order_id and email is checked against access.py first.
"""

import json
from anthropic import beta_tool

from .access import NO_MATCH
from .approvals import RefundArgs, TicketArgs
from .rag import PolicyHit, get_policy_index
from .refunds import RefundReason
from .store import PRODUCTS, find_order
from .tickets import TicketCategory


@beta_tool
def search_products(query: str, max_price: float | None = None) -> str:
    """Search the product catalog by keyword.

    Args:
        query: Words to match against product names, categories, and descriptions, e.g. "hiking boots".
        max_price: Optional highest price in USD to include.
    """
    words = query.lower().split()
    matches = []
    for product in PRODUCTS.values():
        text = f"{product['name']} {product['category']} {product['description']}".lower()
        if not any(word in text for word in words):
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        matches.append({"sku": product["sku"], "name": product["name"], "price": product["price"],
                        "in_stock": product["stock"] > 0})
    if not matches:
        return "No products matched that search."
    return json.dumps(matches)


@beta_tool
def get_product_details(sku: str) -> str:
    """Get full details for one product, including price, stock, and whether it is final sale.

    Args:
        sku: The product's SKU, e.g. "SKU-1001".
    """
    product = PRODUCTS.get(sku.upper())
    if product is None:
        return f"No product found with SKU {sku}."
    return json.dumps(product, ensure_ascii=False)


@beta_tool
def get_order_status(order_id: str, email: str) -> str:
    """Look up a customer's order. Ask the customer for both their order number and the email used to place it.

    Args:
        order_id: The order number, e.g. "ORD-5001".
        email: The email address the order was placed with.
    """
    order = find_order(order_id, email)
    if order is None:
        return NO_MATCH
    return json.dumps(order)


@beta_tool
def search_store_policies(query: str) -> list[PolicyHit]:
    """Search the store's returns, refund, and shipping policies. Use this for questions about returns,
    refunds, exchanges, cancellations, damaged or missing packages, shipping options and costs,
    delivery times, and tracking. Results can be cited.

    Args:
        query: What to look up, in plain words, e.g. "how long do refunds take".
    """
    return get_policy_index().search(query)


@beta_tool
def create_support_ticket(email: str, category: TicketCategory, summary: str, order_id: str | None = None) -> str:
    """Ask for a support ticket for the team. Use this when the customer wants something you can't do
    yourself (start a return, cancel an order, change an address, or report a damaged or missing
    package) and they have confirmed they want a team member to follow up. For refunds, use
    request_refund instead. A team member approves the ticket before it's opened.

    Args:
        email: The customer's email address.
        category: The kind of request.
        summary: A short summary for the support team: what the customer wants, which items, and why.
        order_id: The related order number, if there is one. It must match the customer's email.
    """
    raise _approval_required()


@beta_tool
def request_refund(order_id: str, email: str, reason: RefundReason, skus: list[str] | None = None,
                   details: str = "") -> str:
    """Request a refund once the customer has confirmed they want one. This checks the refund policy:
    if the request qualifies, it's sent to a team member to approve, and you get back the estimated
    amount; if not, you get back why. It never moves money itself.

    Args:
        order_id: The order number, e.g. "ORD-5001".
        email: The email address the order was placed with.
        reason: Why the customer wants a refund.
        skus: The SKUs to refund, e.g. ["SKU-1001"]. Omit for the whole order.
        details: Anything else the support team should know.
    """
    raise _approval_required()


def _approval_required() -> Exception:
    return RuntimeError("Write tools run through the graph's approval gate, not directly.")


ALL_TOOLS = [search_products, get_product_details, get_order_status, search_store_policies, create_support_ticket,
             request_refund]

# Write tools and how to read their arguments as an action (approvals.py).
WRITE_TOOLS = {
    "create_support_ticket": lambda args: TicketArgs(email=args.get("email", ""), category=args.get("category"),
                                                     message=args.get("summary", ""), order_id=args.get("order_id")),
    "request_refund": lambda args: RefundArgs(**args),
}
