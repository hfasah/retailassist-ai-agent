"""Shared input types and output models, so every entry point validates the same way.

Input types (OrderId, Email, ...) normalize first (trim, fix case) and then validate, so "ord-5003 " is
accepted as ORD-5003 but "5003" is rejected with a message the model can act on. The agent's tools,
the MCP server, and the approval queue all use them, and their constraints go into the JSON schemas
the models see.

Output models describe exactly what each tool returns. Building them validates the data, and they
leave out what the model doesn't need (such as the customer's stored contact details).
"""

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, ValidationError


# In the types below, the constraints come before the BeforeValidator so that they stay in the JSON
# schema; the validator still runs first.

def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _upper(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


ORDER_ID_PATTERN = r"^ORD-\d{4,8}$"
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
SKU_PATTERN = r"^SKU-\d{4}$"

OrderId = Annotated[str, StringConstraints(pattern=ORDER_ID_PATTERN), BeforeValidator(_upper),
                    Field(description="Order number, e.g. ORD-5001")]
Email = Annotated[str, StringConstraints(pattern=EMAIL_PATTERN, max_length=254), BeforeValidator(_lower),
                  Field(description="Email address the order was placed with")]
Sku = Annotated[str, StringConstraints(pattern=SKU_PATTERN), BeforeValidator(_upper),
                Field(description="Product SKU, e.g. SKU-1001")]
SearchQuery = Annotated[str, StringConstraints(min_length=1, max_length=200), BeforeValidator(_strip)]
Message = Annotated[str, StringConstraints(min_length=1, max_length=4000), BeforeValidator(_strip)]
Details = Annotated[str, StringConstraints(max_length=2000), BeforeValidator(_strip)]
Price = Annotated[float, Field(gt=0, le=100_000)]


def describe_errors(error: ValidationError, limit: int = 5) -> str:
    """A ValidationError as one short line per field, for a model to read and correct its call."""
    lines = []
    for e in error.errors(include_url=False)[:limit]:
        field = ".".join(str(part) for part in e["loc"]) or "input"
        given = json.dumps(e.get("input"), default=str)
        given = given if len(given) <= 60 else given[:57] + "..."
        message = e["msg"]
        if e["type"] == "string_pattern_mismatch":
            message = f"should look like {_example(e['ctx']['pattern'])}"
        lines.append(f"{field}: {message} (got {given})" if e["type"] != "missing" else f"{field}: required")
    return "; ".join(lines)


def _example(pattern: str) -> str:
    return {ORDER_ID_PATTERN: "ORD-5001", EMAIL_PATTERN: "name@example.com",
            SKU_PATTERN: "SKU-1001"}.get(pattern, f"the pattern {pattern}")


# --- Tool outputs ---------------------------------------------------------------------------------

class Output(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ProductSummary(Output):
    sku: str
    name: str
    price: float
    in_stock: bool


class ProductSearch(Output):
    products: list[ProductSummary]
    message: str = ""


class Product(Output):
    sku: str
    name: str
    category: str
    price: float
    stock: int = Field(ge=0)
    description: str
    final_sale: bool


class OrderItem(Output):
    sku: str
    name: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)


class Order(Output):
    order_id: str = Field(pattern=ORDER_ID_PATTERN)
    order_date: str
    status: Literal["processing", "shipped", "out_for_delivery", "delivered", "return_requested", "refunded",
                    "cancelled"]
    items: list[OrderItem]
    subtotal: float
    shipping_method: str
    shipping_cost: float
    tax: float
    total: float
    carrier: str | None
    tracking_number: str | None
    estimated_delivery: str | None
    delivered_date: str | None
    ship_to: dict[str, str]


class CustomerOrder(Order):
    """An order as the agent sees it: with the customer's name for a friendly reply, but not their email."""
    customer_name: str

    @classmethod
    def from_order(cls, order: dict) -> "CustomerOrder":
        return cls.model_validate(order | {"customer_name": order["customer"]["name"]})


class RefundItem(Output):
    sku: str
    name: str
    quantity: int
    amount: float


class RefundOutcome(Output):
    eligible: bool = Field(description="Whether the request qualifies under the refund policy")
    reasons: list[str] = Field(description="Why it does or doesn't qualify, in plain words")
    items: list[RefundItem] = Field(description="Items included in the refund")
    estimated_refund: float = Field(ge=0, description="Estimated amount in USD, after any return shipping fee")
    return_shipping_fee: float = Field(ge=0)
    ticket_id: str | None = Field(None, description="The refund ticket, once a request has been approved")
    already_requested: bool = Field(False, description="True if a refund request for this order was already open")
    request_id: str | None = Field(None, description="The request waiting for staff approval, if any")
    next_step: str


class ApprovalOutcome(Output):
    """What the agent hears back once staff decide on a write action."""
    request_id: str = Field(pattern=r"^APR-\d+$")
    approved: bool
    note_from_team: str
    message: str = ""
    ticket_id: str | None = None
    status: str | None = None
    estimated_refund: float | None = None
    next_step: str | None = None


def is_valid(pattern: str, value: str | None) -> bool:
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None
