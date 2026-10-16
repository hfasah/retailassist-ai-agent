"""Tools the agent can call.

Every tool's arguments are typed with the shared constraints in schemas.py. The graph validates each
call against them before anything runs (validate_args), and tells the model exactly which field was
wrong so it can fix its call. Tools return output models from schemas.py (or policy hits), which the
graph serializes for the model; building them validates the data.

Read-only tools run directly. Write tools (WRITE_TOOLS) only declare what the model can ask for: the
graph checks order access, gets a person's approval, and then runs the action (see graph.py and
approvals.py). Any tool that takes an order_id and email is checked against access.py first.
"""

import inspect
from functools import cache
from typing import Annotated, Any, get_type_hints

from anthropic import beta_tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .access import NO_MATCH
from .approvals import RefundArgs, TicketArgs
from .rag import PolicyHit, get_policy_index
from .refunds import RefundReason
from .schemas import (CustomerOrder, Details, Email, Message, OrderId, Price, Product, ProductSearch, ProductSummary,
                      SearchQuery, Sku, describe_errors)
from .store import PRODUCTS, find_order
from .tickets import TicketCategory


class ToolInputError(ValueError):
    """A tool call's arguments don't match its schema. The message says which fields, for the model."""


@beta_tool
def search_products(query: Annotated[SearchQuery, Field(description='Words to match against product names, '
                                                                    'categories, and descriptions, e.g. "hiking boots"')],
                    max_price: Annotated[Price | None, Field(description="Optional highest price in USD")] = None
                    ) -> ProductSearch:
    """Search the product catalog by keyword."""
    words = query.lower().split()
    matches = []
    for product in PRODUCTS.values():
        text = f"{product['name']} {product['category']} {product['description']}".lower()
        if not any(word in text for word in words):
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        matches.append(ProductSummary(**product, in_stock=product["stock"] > 0))
    return ProductSearch(products=matches, message="" if matches else "No products matched that search.")


@beta_tool
def get_product_details(sku: Sku) -> Product:
    """Get full details for one product, including price, stock, and whether it is final sale."""
    product = PRODUCTS.get(sku)
    if product is None:
        raise ValueError(f"No product found with SKU {sku}.")
    return Product.model_validate(product)


@beta_tool
def get_order_status(order_id: OrderId, email: Email) -> CustomerOrder:
    """Look up a customer's order. Ask the customer for both their order number and the email used to place it."""
    order = find_order(order_id, email)
    if order is None:
        raise ValueError(NO_MATCH)
    return CustomerOrder.from_order(order)


@beta_tool
def search_store_policies(query: SearchQuery) -> list[PolicyHit]:
    """Search the store's returns, refund, and shipping policies. Use this for questions about returns,
    refunds, exchanges, cancellations, damaged or missing packages, shipping options and costs,
    delivery times, and tracking. Results can be cited.

    Args:
        query: What to look up, in plain words, e.g. "how long do refunds take".
    """
    return get_policy_index().search(query)


@beta_tool
def create_support_ticket(email: Email, category: TicketCategory, summary: Message,
                          order_id: OrderId | None = None) -> str:
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
def request_refund(order_id: OrderId, email: Email, reason: RefundReason,
                   skus: Annotated[list[Sku] | None, Field(max_length=20)] = None, details: Details = "") -> str:
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

TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}

# Write tools and how to read their (already validated) arguments as an action (approvals.py).
WRITE_TOOLS = {
    "create_support_ticket": lambda args: TicketArgs(email=args["email"], category=args["category"],
                                                     message=args["summary"], order_id=args["order_id"]),
    "request_refund": lambda args: RefundArgs(**args),
}


@cache
def _args_model(name: str) -> type[BaseModel]:
    """A Pydantic model of a tool's parameters, built from its signature (the same one its schema comes from)."""
    func = TOOLS_BY_NAME[name].func
    hints = get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hints[param.name], default)
    return create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)


def validate_args(name: str, args: Any) -> dict:
    """Check a tool call's arguments against the tool's schema and return them normalized
    (e.g. "ord-5003 " becomes "ORD-5003"). Raises ToolInputError with a message for the model."""
    if name not in TOOLS_BY_NAME:
        raise ToolInputError(f"Unknown tool: {name}. Available tools: {', '.join(TOOLS_BY_NAME)}.")
    if not isinstance(args, dict):
        raise ToolInputError(f"Arguments for {name} must be an object.")
    try:
        return _args_model(name).model_validate(args).model_dump()
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for {name}: {describe_errors(e)}. Fix them and try again, or ask "
                             "the customer if you need different details.") from e
